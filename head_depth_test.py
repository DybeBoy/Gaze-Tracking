from library import *
import mediapipe as mp
import cv2

# Setup Mediapipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    if key == 32:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb_frame)

        if res.multi_face_landmarks:
            print(get_head_depth(res, frame))
            eye_img = get_eye_input_data(res, frame)
            print(get_head_depth(res, eye_img))