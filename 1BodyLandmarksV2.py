import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# -----------------------------------
# MediaPipe setup
# -----------------------------------
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Define the POSE_CONNECTIONS for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (31, 32)
]

# -----------------------------------
# UI / Toggle settings
# -----------------------------------
show_all_landmarks = True
IMPORTANT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]

WINDOW_NAME = "Pose Landmarks HUD"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# -----------------------------------
# Helper: Draw skeleton connections
# -----------------------------------
def draw_skeleton(frame, landmarks, connections):
    """Draw skeleton lines between connected landmarks"""
    h, w, _ = frame.shape
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            # Only draw if both landmarks are visible (presence > 0.5)
            if hasattr(start_lm, 'presence') and start_lm.presence < 0.5:
                continue
            if hasattr(end_lm, 'presence') and end_lm.presence < 0.5:
                continue
            
            start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
            end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

# -----------------------------------
# Helper: draw top-left HUD panel
# -----------------------------------
def draw_hud_panel(frame, show_all):
    mode_text = "ALL landmarks" if show_all else "IMPORTANT landmarks"
    mode_color = (0, 180, 0) if show_all else (0, 140, 255)

    overlay = frame.copy()

    # Main HUD background
    cv2.rectangle(overlay, (15, 15), (390, 125), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "POSE LANDMARK VIEW", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Mode badge
    cv2.rectangle(frame, (30, 55), (250, 85), mode_color, -1)
    cv2.putText(frame, f"Mode: {mode_text}", (40, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Key hints
    cv2.putText(frame, "[T] Toggle landmarks", (30, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    cv2.putText(frame, "[ESC] Exit", (220, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

# -----------------------------------
# Helper: draw landmark labels
# -----------------------------------
def draw_landmark_labels(frame, landmarks, w, h, show_all, important_ids):
    for idx, lm in enumerate(landmarks):
        px, py = int(lm.x * w), int(lm.y * h)

        # Always draw the landmark point
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        should_show_label = show_all or (idx in important_ids)

        if should_show_label:
            label = str(idx)

            # text size
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # background box
            x1 = px + 6
            y1 = py - 16
            x2 = x1 + tw + 8
            y2 = y1 + th + 8

            # keep inside frame
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 4, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

# -----------------------------------
# Main
# -----------------------------------
cap = cv2.VideoCapture(0)

# Try to increase actual camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("Actual frame size:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create pose landmarker with new API
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose landmarks
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            # Draw skeleton connections
            draw_skeleton(frame, landmarks, POSE_CONNECTIONS)
            
            # Draw landmark points + labels
            draw_landmark_labels(
                frame,
                landmarks,
                w, h,
                show_all_landmarks,
                IMPORTANT_IDS
            )
        
        # Draw HUD last so it stays clean on top
        draw_hud_panel(frame, show_all_landmarks)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('t'):
            show_all_landmarks = not show_all_landmarks

cap.release()
cv2.destroyAllWindows()