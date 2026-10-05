import cv2
import time
import numpy as np

def raw_fps_test(cap, frames=120, width=None, height=None):
    
    if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    time.sleep(1)

    start = time.perf_counter()
    got = 0
    for _ in range(frames):
        ret, _ = cap.read()
        if not ret: break
        got += 1
    elapsed = time.perf_counter() - start
    cap.release()
    return got / elapsed

resolutions = [
    (160, 120),
    (320, 240),
    (352, 288),
    (424, 240),
    (640, 360),
    (640, 480),
    (800, 600),
    (960, 540),
    (1024, 576),
    (1024, 768),
    (1280, 720),
    (1280, 800),
    (1280, 960),
    (1440, 900),
    (1600, 900),
    (1600, 1200),
    (1920, 1080),
    (1920, 1200),
    (2560, 1440),
    (3840, 2160),
]

print("---")

supported = []
fps = []
for width, height in resolutions:
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    break_loop = False
    for i, (s_width, s_height) in enumerate(supported):
            if s_width == actual_width and s_height == actual_height:
                break_loop = True
    
    if break_loop:
        cap.release()
        continue
    
    if actual_width == width and actual_height == height:
        supported.append((width, height))
        print(f"✓ {width}x{height}")

        raw_fps = raw_fps_test(cap, width=width, height=height)
        fps.append(raw_fps)
        print(f"FPS: {raw_fps:.3f}")

    else:
        print(f"✗ {width}x{height} (got {actual_width}x{actual_height})")
        supported.append((actual_width, actual_height))

        raw_fps = raw_fps_test(cap, width=actual_width, height=actual_height)
        fps.append(raw_fps)
        print(f"FPS: {raw_fps:.3f}")

    print("---")

np.save("webcam_resolutions.npy", np.array(supported))
np.save("webcam_fps.npy", np.array(fps))