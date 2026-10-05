import mediapipe as mp
import cv2
from library import *
import time
import numpy as np

supported = []
fps = []

try:
    np.load("webcam_resolutions.npy")
except FileNotFoundError:
    print("No supported resolutions found. Running webcam_resolutions.py to detect supported resolutions.")
    import webcam_resolutions

supported = np.load("webcam_resolutions.npy")
fps = np.load("webcam_fps.npy")

_face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1
)
face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(_face_options)
cap = cv2.VideoCapture(0)

# --- Resolution selection ---
print("\nSupported resolutions:")
for i, (width, height) in enumerate(supported):
    print(f"  {i}: {width}x{height}  ({fps[i]:.3f} FPS)")
while True:
    try:
        answer = int(input("Select resolution (index): "))
        if 0 <= answer < len(supported):
            break
        print(f"  Please enter a number between 0 and {len(supported)-1}.")
    except ValueError:
        print("  Please enter a number.")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, supported[answer][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, supported[answer][1])
cap.set(cv2.CAP_PROP_FPS, fps[answer])

# --- Dataset selection ---
def _list_dir_sorted(path):
    try:
        return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    except FileNotFoundError:
        return []

def _session_sample_count(session_path):
    npz = os.path.join(session_path, "data.npz")
    try:
        with np.load(npz, allow_pickle=True) as d:
            return len(d["image_ids"])
    except Exception:
        return 0

# Step A: dataset type
print("\n--- Step 1: Dataset type ---")
types = _list_dir_sorted("data")
if types:
    for i, t in enumerate(types):
        print(f"  {i}: {t}")
else:
    print("  No dataset types found.")
print(f"  {len(types)}: Create new type")

while True:
    raw = input("Select type (index or name): ").strip()
    if not raw:
        continue
    try:
        idx = int(raw)
        if 0 <= idx < len(types):
            dataset_type_name = types[idx]
            break
        elif idx == len(types):
            dataset_type_name = input("  Enter new type name: ").strip()
            if dataset_type_name:
                break
            print("  Name cannot be empty.")
        else:
            print(f"  Please enter a number between 0 and {len(types)}.")
    except ValueError:
        dataset_type_name = raw
        break

# Step B: session
print(f"\n--- Step 2: Session in '{dataset_type_name}' ---")
sessions = _list_dir_sorted(f"data/{dataset_type_name}")
if sessions:
    for i, s in enumerate(sessions):
        count = _session_sample_count(f"data/{dataset_type_name}/{s}")
        print(f"  {i}: {s}  ({count} samples)")
else:
    print("  No sessions found.")
print(f"  {len(sessions)}: Create new session")

while True:
    raw = input("Select session (index or name): ").strip()
    if not raw:
        continue
    try:
        idx = int(raw)
        if 0 <= idx < len(sessions):
            session_name = sessions[idx]
            break
        elif idx == len(sessions):
            default = f"sesh{len(sessions)+1}"
            entered = input(f"  New session name (press Enter for '{default}'): ").strip()
            session_name = entered if entered else default
            break
        else:
            print(f"  Please enter a number between 0 and {len(sessions)}.")
    except ValueError:
        session_name = raw
        break

dataset_type = f"{dataset_type_name}/{session_name}"
dataset = GazeDataset(root=f"data/{dataset_type}")

# --- Cutoff ---
while True:
    try:
        cutoff = int(input("\nMax samples per target point (0 = no limit): "))
        if cutoff >= 0:
            break
        print("  Please enter 0 or a positive number.")
    except ValueError:
        print("  Please enter a number.")
print()

cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Predetermined normalized screen target positions
TARGETS = [
    # Center
    (0.5, 0.5),

    # Inner ring
    (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),
    (0.25, 0.5),               (0.75, 0.5),
    (0.25, 0.75), (0.5, 0.75), (0.75, 0.75),

    # Outer ring
    (0.1, 0.1),  (0.5, 0.1),  (0.9, 0.1),
    (0.1, 0.5),               (0.9, 0.5),
    (0.1, 0.9),  (0.5, 0.9),  (0.9, 0.9),

    # Extreme corners
    (0.01, 0.01), (0.99, 0.01),
    (0.01, 0.99), (0.99, 0.99),

    #Extreme edges
    (0.5, 0.01), (0.5, 0.99),
    (0.01, 0.5), (0.99, 0.5)
]

# Capture behavior
capture_interval = 0.1    # seconds between frames in continuous mode
target_radius = 18        # screen-drawn radius px
font = cv2.FONT_HERSHEY_SIMPLEX

# --- State ---
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w, h = int(w), int(h)

point_idx = 0
is_continuous = False
last_capture_time = 0.0

counts = [0 for _ in TARGETS]
total_saved = 0

def draw_ui(screen_img, px, py, idx, counts, is_continuous):
    #screen_img[:] = (0, 0, 0)
    # draw other targets faintly
    for i, (nx, ny) in enumerate(TARGETS):
        tx, ty = int(nx * w), int(ny * h)
        if i == idx:
            if is_continuous:
                cv2.circle(screen_img, (tx, ty), int((target_radius+6)*w/1920), (0, 255, 0), 3)
            else:
                cv2.circle(screen_img, (tx, ty), int((target_radius+6)*w/1920), (0, 160, 255), 3)
        else:
            cv2.circle(screen_img, (tx, ty), int(8*w/1920), (60, 60, 60), -1)
    # draw active target
    cv2.circle(screen_img, (px, py), int(target_radius*w/1920), (0, 255, 0), -1)

    if w < 1280:
        return

    # draw instructions
    lines = [
        f"Point {idx+1}/{len(TARGETS)}  samples:{counts[idx]}  total:{sum(counts)}",
        "LEFT/RIGHT or A/D: move points    SPACE: save sample    C: toggle continuous",
        "ESC or Q: quit   Z: clear last 10 samples",
        f"Continuous: {'ON' if is_continuous else 'OFF'}"
    ]
    y0 = 40
    for i, l in enumerate(lines):
        cv2.putText(screen_img, l, (int(20*w/1920), int(y0 + i*28*w/1920)), font, 0.8*w/1920, (200,200,200), 2, cv2.LINE_AA)

while True:
    now = time.time()
    # compute current target pixel coordinates
    nx, ny = TARGETS[point_idx]
    px, py = int(nx * w), int(ny * h)

    ret, frame = cap.read()
    backround_frame = frame.copy()
    if not ret:
        print("Failed to grab frame. Exiting.")
        continue

    # draw UI
    draw_ui(backround_frame, px, py, point_idx, counts, is_continuous)
    cv2.imshow("Frame", backround_frame)

    key = cv2.waitKey(1)

    # Key handling: support several common key codes for arrows across platforms
    LEFT_KEYS = [81, 2424832, 65361, ord('a'), ord('A')]
    RIGHT_KEYS = [83, 2555904, 65363, ord('d'), ord('D')]

    if key != -1:
        # quit
        if key in [27]:
            break
        # left
        if key in LEFT_KEYS:
            point_idx = (point_idx - 1) % len(TARGETS)
            is_continuous = False
            #print(f"Moved to point {point_idx+1}")
            continue
        # right
        if key in RIGHT_KEYS:
            point_idx = (point_idx + 1) % len(TARGETS)
            is_continuous = False
            #print(f"Moved to point {point_idx+1}")
            continue
        # toggle continuous capture
        if key in [ord('c'), ord('C')]:
            is_continuous = not is_continuous
            last_capture_time = 0.0
            #print(f"Continuous capture: {is_continuous}")
            continue
        # clear last 10 samples
        if key in [ord('z'), ord('Z')]:
            for i in range(10):
                dataset.clear_item(dataset.__len__() - 1)
            total_saved -= 10
            counts[point_idx] = max(0, counts[point_idx] - 10)
            #print("Cleared last 10 samples.")
            continue
        # save single sample
        if key == 32:  # SPACE
            # we'll handle saving below by forcing a single capture cycle
            force_single = True
        else:
            force_single = False
    else:
        force_single = False

    # If continuous and enough time passed OR single forced, capture
    do_capture = False
    if force_single:
        do_capture = True
    elif is_continuous and (now - last_capture_time) >= capture_interval:
        do_capture = True

    if not do_capture:
        continue

    if cutoff > 0 and counts[point_idx] >= cutoff:
        #print(f"Cutoff reached for point {point_idx+1}.")
        is_continuous = False
        continue

    last_capture_time = now
    

    key = cv2.waitKey(1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    res = face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    cv2.imshow("Frame", backround_frame)

    if not res.face_landmarks:
        print("No face detected. Skipping frame.")
        continue

    # Reject detections where key landmarks are outside the frame (partially cropped face)
    landmarks = res.face_landmarks[0]
    key_ids = [33, 133, 362, 263, 1, 61, 291, 199]
    if any(not (0.0 <= landmarks[i].x <= 1.0 and 0.0 <= landmarks[i].y <= 1.0) for i in key_ids):
        print("Key landmarks outside frame bounds. Skipping frame.")
        continue

    face_img = get_face_crop(res, frame)
    if face_img is None:
        print("Could not extract face region. Skipping frame.")
        continue

    head_rot = get_head_rotations(res, frame)
    if head_rot is None:
        print("Could not compute head pose. Skipping frame.")
        continue

    head_depth = get_head_depth(res, frame, w)

    head_pos = get_head_position(res)

    iris = get_iris_landmarks(res)

    label = np.array([(nx-0.5)*2, (ny-0.5)*2], dtype=np.float32)

    # Mirror data to get more image per image
    m_face_img = cv2.flip(face_img, 1)

    # Apply independent random channel/brightness augmentation to each copy
    face_img = randomly_increase_channels(face_img)
    m_face_img = randomly_increase_channels(m_face_img)

    m_head_rot = head_rot.copy()
    # 6D rotation (rmat[:, :2].flatten() = [r00,r10,r20, r01,r11,r21])
    # Horizontal flip negates the x-row: indices 0, 2, 3, 5
    m_head_rot[0] = -m_head_rot[0]
    m_head_rot[2] = -m_head_rot[2]
    m_head_rot[3] = -m_head_rot[3]
    m_head_rot[5] = -m_head_rot[5]

    m_head_pos = head_pos.copy()
    m_head_pos[0] = -m_head_pos[0]  # invert x position

    m_head_depth = head_depth.copy()

    # Mirror iris: swap left/right eye and invert x ratios
    m_iris = np.array([-iris[2], -iris[3], -iris[0], -iris[1]], dtype=np.float32)

    m_label = label.copy()
    if m_label[0] != 0: 
        m_label[0] = -m_label[0]  # invert x label

    # Save data
    dataset.save_item(face_img, head_rot, head_pos, head_depth, iris, label)
    dataset.save_item(m_face_img, m_head_rot, m_head_pos, m_head_depth, m_iris, m_label)

    counts[point_idx] += 2
    total_saved += 2

    # brief visual feedback: flash circle
    cv2.circle(backround_frame, (px, py), target_radius+4, (0, 255, 255), 4)
    cv2.imshow("Frame", backround_frame)
    cv2.waitKey(1)
"""
    print(
        label,
        head_pos,
        head_rot,
        head_depth,
        "\n",
        m_label,
        m_head_pos,
        m_head_rot,
        m_head_depth
    )
"""
cap.release()
cv2.destroyAllWindows()
dataset.flush()