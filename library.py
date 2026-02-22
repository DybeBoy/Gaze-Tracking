import mediapipe as mp
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torch
import os
import math

def get_eye_input_data(res, frame):
    landmarks = res.multi_face_landmarks[0].landmark
    eye_ids = [33, 133, 362, 263]
    xs = [landmarks[i].x for i in eye_ids]
    ys = [landmarks[i].y for i in eye_ids]
    
    x_min = int(min(xs) * frame.shape[1] - 12/1920*frame.shape[1])
    x_max = int(max(xs) * frame.shape[1] + 12/1920*frame.shape[1])
    y_min = int(min(ys) * frame.shape[0] - 18/1920*frame.shape[1])
    y_max = int(max(ys) * frame.shape[0] + 18/1920*frame.shape[1])

    # Clamp coords
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)

    # Validate coords
    if x_min >= x_max or y_min >= y_max:
        return None

    eye_img = frame[y_min:y_max, x_min:x_max]
    if eye_img.size == 0:
        return None
    
    eye_img = cv2.resize(eye_img, (96, 96))
    #eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)

    return eye_img


def get_head_position(res):
    landmarks = res.multi_face_landmarks[0].landmark
    eye_ids = [33, 263]

    x_positions = [landmarks[i].x for i in eye_ids]
    y_positions = [landmarks[i].y for i in eye_ids]
    head_pos = [((x_positions[0] + x_positions[1]) / 2 - 0.5) * 2,
                ((y_positions[0] + y_positions[1]) / 2 - 0.5) * 2]

    return np.array(head_pos, dtype=np.float32)

def get_head_rotations(res, frame):
    landmarks = res.multi_face_landmarks[0].landmark

    FACE_IDS = {
        "nose_tip": 1,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth": 61,
        "right_mouth": 291,
        "chin": 199
    }

    model_points_3d = np.array([
        [0.0, 0.0, 0.0],
        [-30.0, -125.0, -30.0],
        [30.0, -125.0, -30.0],
        [-60.0, -70.0, -60.0],
        [60.0, -70.0, -60.0],
        [0.0, -150.0, -10.0]
    ], dtype=np.float64)

    h, w = frame.shape[:2]
    image_points_2d = np.array([
        [landmarks[FACE_IDS["nose_tip"]].x * w, landmarks[FACE_IDS["nose_tip"]].y * h],
        [landmarks[FACE_IDS["left_eye_outer"]].x * w, landmarks[FACE_IDS["left_eye_outer"]].y * h],
        [landmarks[FACE_IDS["right_eye_outer"]].x * w, landmarks[FACE_IDS["right_eye_outer"]].y * h],
        [landmarks[FACE_IDS["left_mouth"]].x * w, landmarks[FACE_IDS["left_mouth"]].y * h],
        [landmarks[FACE_IDS["right_mouth"]].x * w, landmarks[FACE_IDS["right_mouth"]].y * h],
        [landmarks[FACE_IDS["chin"]].x * w, landmarks[FACE_IDS["chin"]].y * h],
    ], dtype=np.float64)

    focal_length = (w + h) / 2.0
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(model_points_3d, image_points_2d, camera_matrix, dist_coeffs)
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    # Return normalized angles as float32. Scaling by 45 keeps values in a reasonable [-1,1] range for training.
    return np.array([math.degrees(yaw), math.degrees(pitch), math.degrees(roll)], dtype=np.float32) / 45.0


def get_head_depth(res, frame, base_img_width):
    landmarks = res.multi_face_landmarks[0].landmark
    eye_ids = [33, 263]

    x_positions = [landmarks[i].x for i in eye_ids]
    y_positions = [landmarks[i].y for i in eye_ids]

    iod_px_x = np.linalg.norm(x_positions[1] - x_positions[0])
    iod_px_y = np.linalg.norm(y_positions[1] - y_positions[0])
    iod_px = math.sqrt(iod_px_x**2 + iod_px_y**2) / frame.shape[1]

    iod_px *= 10000000.0  # Scale to a more manageable range
    iod_px /= base_img_width  # Normalize by base image width

    return np.array([iod_px], dtype=np.float32)

# Reusable inference transform to avoid recreating it every frame
infer_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

prev_rot = None
def gaze_prediction(model, frame, device, face_mesh=None):
    global prev_rot
    head_scale = model.head_scale

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    created_local = False
    if face_mesh is None:
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(refine_landmarks=True)
        created_local = True

    res = face_mesh.process(rgb_frame)

    if not res.multi_face_landmarks:
        if created_local:
            face_mesh.close()
        return None

    eye_data = get_eye_input_data(res, frame)
    if eye_data is None:
        if created_local:
            face_mesh.close()
        return None
    
    head_rot = get_head_rotations(res, frame) * head_scale
    if head_rot is None:
        if created_local:
            face_mesh.close()
        return None
    
    alpha = 0.05
    if prev_rot is not None:
        head_rot = alpha * head_rot + (1 - alpha) * prev_rot
        prev_rot = head_rot
    else:
        prev_rot = head_rot

    head_pos = get_head_position(res) * head_scale

    head_depth = get_head_depth(res, frame, frame.shape[1]) * head_scale

    head_input = np.concatenate([head_rot, head_pos, head_depth]).astype(np.float32)

    # Convert OpenCV BGR numpy array to PIL RGB image for torchvision transforms
    eye_pil = Image.fromarray(cv2.cvtColor(eye_data, cv2.COLOR_BGR2RGB))
    eye_tensor = infer_transform(eye_pil).unsqueeze(0).to(device)
    
    head_input_tensor = torch.tensor(head_input).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(eye_tensor, head_input_tensor)

    gaze = prediction.cpu().numpy()

    if created_local:
        face_mesh.close()

    return {
        "x": gaze[0][0],
        "y": gaze[0][1]
    }


def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, x))


def randomly_increase_channels(image, channel_range=(1.0, 1.3)):
    """
    image: numpy array with shape (H, W, 3) or (H, W, 4)
    channel_range: tuple of (min, max) multiplication factors
    """
    image = image.astype(float)  # Convert to float for processing
    
    # Generate random scale for each channel
    scales = np.random.uniform(channel_range[0], channel_range[1], size=3)
    
    # Apply scales to each channel
    image[:, :, :3] = image[:, :, :3] * scales
    
    # Clip values to valid range (e.g., 0-255)
    image = np.clip(image, 0, 255)
    
    return image.astype(np.uint8)


def get_files_from_root(root):
    folders = []
    for f in os.listdir(root):
        if f != "images" and f!= "data.npz":
            folders.append(root + "/" + f)
    print("Folders: ", folders, "\n")

    if len(folders) == 0:
        folders.append(root)

    data_files = []
    for i in folders:
        for f in os.listdir(i):
            if os.path.isfile(os.path.join(i, f)):
                if f[-3:] == "npz":
                    data_files.append(i + "/" + f)
    print("Data files: ", data_files, "\n")

    image_folders = []
    for i in folders:
        for f in os.listdir(i):
            if f == "images":
                image_folders.append(i + "/" + f)
    print("Image folders: ", image_folders, "\n")

    if folders[0] == root:
        folders = []

    return folders, data_files, image_folders


class GazeDataset(Dataset):
    def __init__(self, root, transform=None, head_scale=1.0):
        self.root = root
        self.transform = transform
        self.head_scale = head_scale
        try:
            folders, data_files, image_folders = get_files_from_root(root)

            data = 0

            if data_files == []:
                data = np.load(root + "/data.npz", allow_pickle=True)
            else:
                with np.load(data_files[0], allow_pickle=True) as tmp:
                    keys = list(tmp.keys())

                acc = {k: [] for k in keys}
                for f in data_files:
                    with np.load(f) as data:
                        for k in keys:
                            acc[k].append(data[k])

                data = {k: np.concatenate(acc[k], axis=0) for k in keys}

            self.image_ids = data["image_ids"]
            self.head_rots = data["head_rots"]
            self.head_pos = data["head_pos"]
            self.head_depths = data["head_depths"]
            self.labels = data["labels"]
            
        except FileNotFoundError:
            self.image_ids = np.array([])
            self.head_rots = np.array([])
            self.head_pos = np.array([])
            self.head_depths = np.array([])
            self.labels = np.array([])

            os.makedirs(f"{self.root}/images", exist_ok=True)
            print("Initialized new dataset.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_folder_name, _, image_id = str.partition(self.image_ids[idx], "_")
        img_path = f"{self.root}/{img_folder_name}/images/{image_id}"
        image = Image.open(img_path).convert("RGB") 

        if self.transform:
            image = self.transform(image)

        head_scale = self.head_scale

        head_rot = torch.tensor(self.head_rots[idx], dtype=torch.float32) * head_scale
        head_pos = torch.tensor(self.head_pos[idx], dtype=torch.float32) * head_scale
        head_depth = torch.tensor(self.head_depths[idx], dtype=torch.float32) * head_scale
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, head_rot, head_pos, head_depth, label

    def save_item(self, img, head_rot, head_pos, head_depth, label):
        new_root = str.partition(self.root, "/")[2]
        last_root_folder = str.partition(new_root, "/")[2]
        img_name = f"{len(self.image_ids):07d}.png"
        img_id = f"{last_root_folder}_{len(self.image_ids):07d}.png"
        cv2.imwrite(f"{self.root}/images/{img_name}", img)
        
        self.image_ids = np.append(self.image_ids, img_id)

        self.head_rots = (
            np.vstack([self.head_rots, head_rot])
            if self.head_rots.size else np.array([head_rot])
        )

        self.head_pos = (
            np.vstack([self.head_pos, head_pos])
            if self.head_pos.size else np.array([head_pos])
        )

        self.head_depths = (
            np.vstack([self.head_depths, head_depth])
            if self.head_depths.size else np.array([head_depth])
        )

        self.labels = (
            np.vstack([self.labels, label])
            if self.labels.size else np.array([label])
        )

        np.savez(
            f"{self.root}/data.npz",
            image_ids=self.image_ids,
            head_rots=self.head_rots,
            head_pos=self.head_pos,
            head_depths=self.head_depths,
            labels=self.labels
        )

    def clear(self):
        # Get user confirmation
        confirm = input("Are you sure you want to clear the dataset? This action cannot be undone. (y/n): ")
        if confirm.lower() != 'y':
            print("Clear operation cancelled.")
            return False
        print("Clearing dataset...")

        # Reset data arrays
        self.image_ids = np.array([])
        self.head_rots = np.array([])
        self.head_pos = np.array([])
        self.head_depths = np.array([])
        self.labels = np.array([])

        folders, data_files, image_folders = get_files_from_root(self.root)

        # Remove images
        for img_folder in image_folders:
            for filename in os.listdir(img_folder):
                file_path = os.path.join(img_folder, filename)
                os.remove(file_path)

        # Remove data file
        for data_file in data_files:
            if os.path.exists(data_file):
                os.remove(data_file)

        print("Dataset cleared.")
        return True
    
    # Only use on the last index so that the saving doesnt go badonkadonks
    def clear_item(self,idx):
        if idx < 0 or idx >= len(self.image_ids):
            print("Index out of range. Cannot clear item.")
            return False

        # Remove image file
        img_id = self.image_ids[idx]
        img_path = f"{self.root}/images/{img_id}"
        if os.path.exists(img_path):
            os.remove(img_path)

        # Remove data from arrays
        self.image_ids = np.delete(self.image_ids, idx, axis=0)
        self.head_rots = np.delete(self.head_rots, idx, axis=0)
        self.head_pos = np.delete(self.head_pos, idx, axis=0)
        self.head_depths = np.delete(self.head_depths, idx, axis=0)
        self.labels = np.delete(self.labels, idx, axis=0)

        # Save updated data
        np.savez(
            f"{self.root}/data.npz",
            image_ids=self.image_ids,
            head_rots=self.head_rots,
            head_pos=self.head_pos,
            head_depth=self.head_depths,
            labels=self.labels
        )

        return True
    

# Inputs: "eye image"(3, 96, 96), "head position, rotations and depth"(6,)
class NeuralNetworkModel(nn.Module):
    def __init__(self, freeze=True, head_scale=1.0):
        super().__init__()

        self.head_scale = head_scale

        base = models.mobilenet_v2(weights="DEFAULT")
        self.backbone = base.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        DROPOUT_EXTRA = 1.0

        self.visual_fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35 * DROPOUT_EXTRA),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.35 * DROPOUT_EXTRA),
        )

        self.pose_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2 * DROPOUT_EXTRA),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2 * DROPOUT_EXTRA),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4 * DROPOUT_EXTRA),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3 * DROPOUT_EXTRA),

            nn.Linear(64, 2)
        )

        if freeze:
            self.freeze_backbone()

    def forward(self, image, head):
        x = self.backbone(image)
        x = self.pool(x).flatten(1)
        x = self.visual_fc(x)

        h = self.pose_fc(head)

        fused = torch.cat([x, h], dim=1)
        return self.fusion(fused)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks=3):
        for param in self.backbone.parameters():
            param.requires_grad = False

        blocks = [m for m in self.backbone.children()]
        for block in blocks[-last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True