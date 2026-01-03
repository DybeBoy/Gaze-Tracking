import os
import numpy as np
from library import *

dir_path = "files/training/sesh4"

try:
    folders = []
    for f in os.listdir(dir_path):
        if not os.path.isfile(os.path.join(dir_path, f)):
            folders.append(dir_path + "/" + f)
    print(folders)

    files = []
    for i in folders:
        for f in os.listdir(i):
            if os.path.isfile(os.path.join(i, f)):
                files.append(i + "/" + f)
    print(files)

    data = 0

    if files == []:
        print("using dir_path")
        data = np.load(dir_path + "/data.npz", allow_pickle=True)
    else:
        print("using files")
        with np.load(files[0]) as tmp:
            keys = list(tmp.keys())

        acc = {k: [] for k in keys}
        for f in files:
            with np.load(f) as data:
                for k in keys:
                    acc[k].append(data[k])

        data = {k: np.concatenate(acc[k], axis=0) for k in keys}

    print(len(data["image_ids"]))

except FileNotFoundError:
    os.makedirs(dir_path + "/images", exist_ok=True)
    print("folder made")


class GazeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        try:
            data = np.load(f"{root}/data.npz", allow_pickle=True)

            self.image_ids = data["image_ids"]
            self.head_rots = data["head_rots"]
            self.head_pos = data["head_pos"]
            self.head_depths = data["head_depth"]
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
        img_path = f"{self.root}/images/{self.image_ids[idx]}"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        HEAD_SCALE = 0.25

        head_rot = torch.tensor(self.head_rots[idx], dtype=torch.float32) * HEAD_SCALE
        head_pos = torch.tensor(self.head_pos[idx], dtype=torch.float32) * HEAD_SCALE
        head_depth = torch.tensor(self.head_depths[idx], dtype=torch.float32) * HEAD_SCALE
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, head_rot, head_pos, head_depth, label

    def save_item(self, img, head_rot, head_pos, head_depth, label):
        img_id = f"{len(self.image_ids):07d}.png"
        cv2.imwrite(f"{self.root}/images/{img_id}", img)

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
            head_depth=self.head_depths,
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

        # Remove images
        img_folder = f"{self.root}/images"
        for filename in os.listdir(img_folder):
            file_path = os.path.join(img_folder, filename)
            os.remove(file_path)

        # Remove data file
        data_file = f"{self.root}/data.npz"
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