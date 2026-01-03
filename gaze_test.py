import time
import torch
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from library import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("saved models", exist_ok=True)
saved_models = os.listdir("saved models")

model_num = ""

print(f'\nGive the models index (0-{len(saved_models)}). For the newest use "-1".')
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

    if answer == -1:
        model_num = len(saved_models) - 1
    else:
        model_num = answer
    break

model = NeuralNetworkModel().to(device)
model.load_state_dict(torch.load(f"saved models/gaze_model{model_num}.pth", map_location=device))
model.eval()

# Screen size
w, h = pyautogui.size()
screen = np.zeros((h, w, 3), dtype=np.uint8)

# Setup Mediapipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("Gaze Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_delay = 1.0 / 10 # Target 10 FPS
last_frame_time = time.time()

while True:
    current_time = time.time()
    if current_time - last_frame_time < frame_delay:
        continue
    last_frame_time = current_time

    ret, frame = cap.read()
    if not ret:
        continue
    
    gaze_data = gaze_prediction(model, frame, device)

    if gaze_data == None:
        continue

    posX, posY = int(clamp01(gaze_data["x"]) * w), int(clamp01(gaze_data["y"]) * h)

    screen[:] = (0, 0, 0)
    cv2.circle(screen, (posX, posY), 15, (255, 0, 0), -1)
    cv2.imshow("Gaze Detection", screen)
    print(posX/w, posY/h)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()