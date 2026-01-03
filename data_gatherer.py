import mediapipe as mp
import cv2
from library import *
import pyautogui
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam. Exiting.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("\nGive file location for the data")
dataset_type = input("Input: ")
dataset = GazeDataset(root=f"data/{dataset_type}")
dataset.clear()
exit()
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Predetermined normalized screen target positions (13 points)
TARGETS = [
    (0.5, 0.5),
    (0.25, 0.25), (0.75, 0.25),
    (0.25, 0.75), (0.75, 0.75),
    (0.01, 0.01), (0.99, 0.01), (0.01, 0.99), (0.99, 0.99),
    (0.5, 0.01), (0.5, 0.99), (0.01, 0.5), (0.99, 0.5)
]

# Capture behavior
capture_interval = 0.05    # seconds between frames in continuous mode
target_radius = 18        # screen-drawn radius px
font = cv2.FONT_HERSHEY_SIMPLEX

# --- State ---
w, h = pyautogui.size()

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
                cv2.circle(screen_img, (tx, ty), target_radius+6, (0, 255, 0), 3)
            else:
                cv2.circle(screen_img, (tx, ty), target_radius+6, (0, 160, 255), 3)
        else:
            cv2.circle(screen_img, (tx, ty), 8, (60, 60, 60), -1)

    # draw active target
    cv2.circle(screen_img, (px, py), target_radius, (0, 255, 0), -1)

    # draw instructions
    lines = [
        f"Point {idx+1}/{len(TARGETS)}  samples:{counts[idx]}  total:{sum(counts)}",
        "LEFT/RIGHT or A/D: move points    SPACE: save sample    C: toggle continuous",
        "ESC or Q: quit   Z: clear last 10 samples",
        f"Continuous: {'ON' if is_continuous else 'OFF'}"
    ]
    y0 = 40
    for i, l in enumerate(lines):
        cv2.putText(screen_img, l, (20, y0 + i*28), font, 0.8, (200,200,200), 2, cv2.LINE_AA)

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
        if key in [27, ord('q'), ord('Q')]:
            break
        # left
        if key in LEFT_KEYS:
            point_idx = (point_idx - 1) % len(TARGETS)
            is_continuous = False
            print(f"Moved to point {point_idx+1}")
            continue
        # right
        if key in RIGHT_KEYS:
            point_idx = (point_idx + 1) % len(TARGETS)
            is_continuous = False
            print(f"Moved to point {point_idx+1}")
            continue
        # toggle continuous capture
        if key in [ord('c'), ord('C')]:
            is_continuous = not is_continuous
            last_capture_time = 0.0
            print(f"Continuous capture: {is_continuous}")
            continue
        # clear last 10 samples
        if key in [ord('z'), ord('Z')]:
            for i in range(10):
                dataset.clear_item(dataset.__len__() - 1)
            total_saved -= 10
            counts[point_idx] = max(0, counts[point_idx] - 10)
            print("Cleared last 10 samples.")
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

    last_capture_time = now
    

    key = cv2.waitKey(1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb_frame)

    cv2.imshow("Frame", backround_frame)

    if not res.multi_face_landmarks: 
        print("No face detected. Skipping frame.")
        continue

    eye_img = get_eye_input_data(res, frame)
    if eye_img is None:
        print("Could not extract eye region. Skipping frame.")
        continue

    head_rot = get_head_rotations(res, frame)
    if head_rot is None:
        print("Could not compute head pose. Skipping frame.")
        continue

    head_depth = get_head_depth(res, frame)

    head_pos = get_head_position(res)

    label = np.array([(nx-0.5)*2, (ny-0.5)*2], dtype=np.float32)

    # Mirror data to get more image per image
    m_eye_img = cv2.flip(eye_img, 1)

    m_head_rot = head_rot.copy()
    m_head_rot[0] = -m_head_rot[0]  # invert yaw
    m_head_rot[2] = -m_head_rot[2]  # invert roll

    m_head_pos = head_pos.copy()
    m_head_pos[0] = -m_head_pos[0]  # invert x position

    m_head_depth = head_depth.copy()

    m_label = label.copy()
    if m_label[0] != 0: 
        m_label[0] = -m_label[0]  # invert x label

    # Save data
    dataset.save_item(eye_img, head_rot, head_pos, head_depth, label)
    dataset.save_item(m_eye_img, m_head_rot, m_head_pos, m_head_depth, m_label)

    counts[point_idx] += 2
    total_saved += 2

    # brief visual feedback: flash circle
    cv2.circle(backround_frame, (px, py), target_radius+4, (0, 255, 255), 4)
    cv2.imshow("Frame", backround_frame)
    cv2.waitKey(1)

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

cap.release()
cv2.destroyAllWindows()