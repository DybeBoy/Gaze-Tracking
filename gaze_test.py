import time
import torch
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from library import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_model_extra", exist_ok=True)
saved_models = os.listdir("saved_models")

model_num = ""

print(f'\nGive the models index (0-{len(saved_models)-1}). Works from last using "-".')
while True:
    answer = input("Input: ")

    if answer == "fuck no":
        print("ok")
        exit()

    try:
        answer = int(answer)
    except ValueError:
        print("Invalid input.")
        continue

    if answer > len(saved_models) - 1:
        print("Invalid model index")
        continue

    if answer < 0:
        model_num = len(saved_models) + answer
    elif answer >= 0:
        model_num = answer
    break

try:
    head_scale = float(np.load(f"saved_model_extra/head_scale{model_num}.npy")[0])
    val_loss = float(np.load(f"saved_model_extra/head_scale{model_num}.npy")[1])
    print(f"Loaded head scale: {head_scale}, Validation loss: {val_loss}")
except FileNotFoundError:
    head_scale = 1.0

model = NeuralNetworkModel(head_scale=head_scale).to(device)
model.load_state_dict(torch.load(f"saved_models/gaze_model{model_num}.pth", map_location=device))
model.eval()

# Screen size
width, height = pyautogui.size()
screen = np.zeros((height, width, 3), dtype=np.uint8)

# Setup Mediapipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Webcam
supported = []
fps = []

try:
    np.load("webcam_resolutions.npy")
except FileNotFoundError:
    print("No supported resolutions found. Running webcam_resolutions.py to detect supported resolutions.")
    import webcam_resolutions

supported = np.load("webcam_resolutions.npy")
fps = np.load("webcam_fps.npy")

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(1)

print("\nSupported resolutions:")
for i, (w, h) in enumerate(supported):
    print(f" {i}: {w}x{h}, {fps[i]:.3f} FPS")
answer = int(input("Select resolution index: "))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, supported[answer][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, supported[answer][1])
cap.set(cv2.CAP_PROP_FPS, fps[answer])

cv2.namedWindow("Gaze Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
"""
calib_points = [
    (-0.8, -0.8),   (0, -0.8),   (0.8, -0.8),
    (-0.8,  0  ),   (0,  0  ),   (0.8,  0  ),
    (-0.8,  0.8),   (0,  0.8),   (0.8,  0.8),
]
"""
calib_points = [
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

for n, (nx, ny) in enumerate(calib_points):
    calib_points[n] = ((nx - 0.5) * 2, (ny - 0.5) * 2)

preds = []
targets = []

is_continuous = False
last_capture_time = 0.0
capture_interval = 0.12

for (nx, ny) in calib_points:
    px, py = int((nx / 2 + 0.5) * width), int((ny / 2 + 0.5) * height)

    images = 0

    while True:
        now = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        screen = frame.copy()
        screen = cv2.resize(screen, (width, height))
        cv2.circle(screen, (px, py), 15, (0, 255, 0), -1)
        cv2.imshow("Gaze Detection", screen)

        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
        if key in [ord('c'), ord('C')]:
            is_continuous = not is_continuous
            last_capture_time = 0.0
            #print(f"Continuous capture: {is_continuous}")
            continue
        if key == 32 and res.multi_face_landmarks:
                force_single = True
        else:
            force_single = False

        do_capture = False
        if force_single:
            do_capture = True
        elif is_continuous and (now - last_capture_time) >= capture_interval:
            do_capture = True

        if not do_capture:
            continue

        last_capture_time = now

        landmarks = res.multi_face_landmarks[0].landmark
        pred = gaze_prediction(model, frame, device, face_mesh)

        preds.append([pred["x"], pred["y"]])
        targets.append([nx, ny])

        images += 1

        if images >= 1:
            is_continuous = False
            images = 0
            #print("changing")
            break

x = np.hstack([preds, np.ones((len(preds), 1))])
y = np.array(targets)

w, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

frame_delay = 1.0 / 30 # Target 10 FPS
last_frame_time = time.time()

previous_x, previous_y = 0.0, 0.0
alpha = 0.14

adaptive_min_alpha = 0.05
adaptive_max_alpha = 0.85
speed_scale = 50.0

deadzone = 0.001

try:
    while True:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
        last_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            continue
        
        gaze_data = gaze_prediction(model, frame, device, face_mesh)

        if gaze_data is None:
            continue

        rawX, rawY = np.array([gaze_data["x"], gaze_data["y"], 1.0]) @ w
        rawX, rawY = int(clamp((rawX) / 2 + 0.5) * width), int(clamp((rawY) / 2 + 0.5) * height)

        smoothedX, smoothedY = np.array([gaze_data["x"], gaze_data["y"], 1.0]) @ w
        smoothedX, smoothedY = clamp((smoothedX) / 2 + 0.5), clamp((smoothedY) / 2 + 0.5)

        # Smooth the gaze
        if previous_x == 0.0 and previous_y == 0.0:
            previous_x, previous_y = smoothedX, smoothedY
        else:
            smoothedX = alpha * smoothedX + (1 - alpha) * previous_x
            smoothedY = alpha * smoothedY + (1 - alpha) * previous_y

        # Secondary smooth
        speed = math.hypot(rawX - previous_x, rawY - previous_y)
        alpha_adaptive = clamp(
            adaptive_min_alpha + speed * speed_scale,
            adaptive_min_alpha,
            adaptive_max_alpha
        )

        smoothedX = alpha_adaptive * smoothedX + (1 - alpha_adaptive) * previous_x
        smoothedY = alpha_adaptive * smoothedY + (1 - alpha_adaptive) * previous_y

        if abs(smoothedX - previous_x) < deadzone:
            smoothedX = previous_x
        if abs(smoothedY - previous_y) < deadzone:
            smoothedY = previous_y

        previous_x, previous_y = smoothedX, smoothedY

        smoothedX, smoothedY = int(smoothedX * width), int(smoothedY * height)

        screen = frame.copy()
        screen = cv2.resize(screen, (width, height))
        cv2.circle(screen, (rawX, rawY), 10, (155, 55, 0), -1)
        cv2.circle(screen, (smoothedX, smoothedY), 15, (255, 0, 0), -1)
        cv2.imshow("Gaze Detection", screen)
        
        if cv2.waitKey(1) == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()