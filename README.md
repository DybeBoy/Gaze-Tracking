# Gaze Tracking

A real-time eye gaze tracking system that predicts where you are looking on screen using a webcam. Includes a 24-point calibration step and adaptive smoothing for stable output.

## What it does

Captures webcam frames, extracts facial landmarks and head pose information, and feeds them into a small neural network that outputs 2D gaze coordinates on the screen. A calibration routine corrects for per-user variation, and a two-stage adaptive EMA filter reduces jitter.

## What it uses

- **PyTorch**: small neural network with a MobileNetV2 visual backbone fused with a head-pose pathway; outputs screen gaze coordinates
- **MediaPipe**: face mesh detection (468 landmarks) used to extract head rotation, position, depth, and iris positions
- **OpenCV**: webcam capture, frame processing, and visualization
- **NumPy**: data storage and numerical operations

## Neural network

This is a small neural network project. The model has two input pathways, one for the cropped face image (MobileNetV2 backbone) and one for 13-dimensional head pose features, which are fused and passed through a small fully connected head to produce (x, y) gaze coordinates. Training uses progressive fine-tuning across three stages.

## Scripts

| Script | Purpose |
|---|---|
| `data_gatherer.py` | Collect training data via on-screen calibration grid |
| `training.py` | Train the gaze model on collected data |
| `gaze_test.py` | Run real-time gaze tracking with calibration and smoothing |
