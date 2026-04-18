# 🦾 Body Landmarks HUD

A real-time human pose landmark detection app using **MediaPipe** and **OpenCV**. Displays a live skeleton overlay with labeled body landmarks on your webcam feed — with a clean HUD interface.

---

## 📸 Features

- 🎥 Real-time webcam pose detection at **1280×720**
- 🦴 Full **skeleton drawing** with joint connections
- 🏷️ Toggleable landmark **ID labels** (all 33 or key joints only)
- 🖥️ On-screen **HUD panel** with mode indicator and key hints
- ⌨️ Keyboard controls for live interaction

---

## 🧠 Landmark Modes

| Mode | Description |
|------|-------------|
| **ALL** | Shows all 33 MediaPipe pose landmarks |
| **IMPORTANT** | Shows only key joints: nose, shoulders, elbows, wrists, hips (IDs: 0, 11–16, 23, 24) |

Press **`T`** to toggle between modes at runtime.

---

## 🗂️ Project Structure

```
.
├── 1BodyLandmarksV2.py     # Main application script
├── pose_landmarker.task    # MediaPipe pose landmarker model file (required)
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- OpenCV
- MediaPipe

Install dependencies:

```bash
pip install opencv-python mediapipe
```

---

## 📥 Model File

Download the **MediaPipe Pose Landmarker** model and place it in the project root:

```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task -O pose_landmarker.task
```

> The model file (`pose_landmarker.task`) must be in the **same directory** as the script.

---

## 🚀 Usage

```bash
python 1BodyLandmarksV2.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `T` | Toggle between ALL / IMPORTANT landmark labels |
| `ESC` | Exit the application |

---

## 🔗 Pose Landmark Reference

MediaPipe detects **33 body landmarks**. The important ones highlighted in this app:

| ID | Landmark |
|----|----------|
| 0  | Nose |
| 11 | Left Shoulder |
| 12 | Right Shoulder |
| 13 | Left Elbow |
| 14 | Right Elbow |
| 15 | Left Wrist |
| 16 | Right Wrist |
| 23 | Left Hip |
| 24 | Right Hip |

Full landmark map: [MediaPipe Pose Landmarks](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

---

## 🛠️ Built With

- [MediaPipe](https://developers.google.com/mediapipe) — Pose Landmarker Task API
- [OpenCV](https://opencv.org/) — Video capture and rendering

---

## 📄 License

MIT License — free to use, modify, and distribute.
