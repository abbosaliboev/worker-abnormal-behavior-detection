# Worker Abnormal Behavior Detection

A real-time rule-based system for detecting abnormal worker behaviors using pose estimation and object tracking. No model training required — pure biomechanical rules applied to YOLO11s-pose keypoints.

**[한국어](README_KO.md) | [O'zbek](README_UZ.md)**

---

## Detected Behaviors

| Behavior | Description | Accuracy |
|---|---|---|
| **Fall Detection** | Detects when a worker falls suddenly | 92.4% |
| **Unsafe Running** | Detects running in restricted/dangerous zones | 91.0% |
| **Long-time Inactivity** | Detects workers motionless for 5+ minutes | 95.8% |

> Evaluated using Leave-One-Out Cross-Validation (LOOCV) across subjects.

---

## Project Structure

```
worker-abnormal-behavior-detection/
│
├── fall_detection/               # Fall Detection module
│   ├── detector.py               # Detection logic (rules)
│   └── evaluate.py               # Evaluation script
│
├── running_detection/            # Unsafe Running module
│   ├── detector.py               # Detection logic (rules)
│   └── evaluate.py               # Evaluation script
│
├── inactivity_detection/         # Long-time Inactivity module
│   ├── detector.py               # Detection logic (rules)
│   └── evaluate.py               # Evaluation script
│
├── src/                          # Shared core modules
│   ├── config.py                 # All thresholds and settings
│   ├── pose_extractor.py         # YOLO11s-pose + ByteTracker
│   ├── feature_extractor.py      # Biomechanical feature computation
│   └── behavior_monitor.py       # Orchestrates all three detectors
│
├── datasets/                     # Dataset utilities
│   ├── npy_loader.py             # Load pre-extracted keypoints (X.npy)
│   └── download_running.py       # Download KTH Action dataset
│
├── evaluation/
│   └── feature_utils.py          # Shared feature extraction helper
│
├── main.py                       # Real-time demo entry point
├── requirements.txt
├── REPORT.md                     # Detailed technical report
└── README.md / README_UZ.md / README_KO.md
```

---

## How It Works

```
CCTV / Camera
      ↓
YOLO11s-pose  →  17 body keypoints per person
      ↓
ByteTracker   →  Unique ID assigned to each worker
      ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ fall_detection/  │running_detection/│inactivity_       │
│ detector.py      │ detector.py      │detection/        │
│                  │                  │ detector.py      │
└──────────────────┴──────────────────┴──────────────────┘
      ↓
Alert  (FALL | RUNNING | INACTIVITY)
```

### Fall Detection Logic

**Step 1 — Measure body tilt angle:** Every frame, the system calculates how far the person's spine has tilted from vertical (0° = upright, 90° = horizontal).

**Step 2 — Measure tilt speed:** The system checks how fast that angle is changing (degrees per second).

**Decision:** If the body is tilted more than 70° AND the tilt happened faster than 65°/sec → **FALL detected**

> Why speed matters: a real fall happens in 0.3–0.5 seconds (rapid). Deliberately lying down takes 3–5 seconds (slow). Speed separates the two.

---

### Unsafe Running Logic

**Step 1 — Track horizontal movement:** Every frame, the system measures how fast the person's center of mass moves horizontally across the camera frame.

**Step 2 — Calibrate per environment:** The threshold is calibrated on training subjects to fit the specific camera setup.

**Decision:** If horizontal speed exceeds the calibrated threshold → **RUNNING detected**

> Running moves ~2× faster than walking in the camera frame.

---

### Inactivity Logic

**Step 1 — Count still frames:** The system checks what fraction of recent frames have near-zero body joint movement.

**Step 2 — Check posture stability:** It also checks whether the body angle is stable (a person picking up objects bends and straightens — excluded).

**Step 3 — Start timer:** If both conditions hold, a 5-minute countdown begins. Any movement resets the timer to zero.

**Decision:** If the worker stays motionless for 5 continuous minutes → **INACTIVITY alert**

---

## Results

| Detector | Accuracy | Dataset | Protocol |
|---|---|---|---|
| Fall | **92.40%** ± 3.4% | UP-Fall (4 subjects) | LOOCV |
| Running | **90.99%** ± 0.4% | KTH Action (25 subjects) | LOOCV |
| Inactivity | **95.83%** ± 4.2% | UP-Fall (4 subjects) | LOOCV |
| **Average** | **93.07%** | | |

---

## Datasets

### UP-Fall Detection Dataset
- **Source:** Martinez-Velasco et al., *Data* 2019 — [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/)
- **Subjects used:** 4 out of 17 (Subjects 1–4)
- **Camera:** RGB, ~17 fps, indoor
- **Total windows:** 4,479 (30 frames each, stride 15)
- **Activities used:**

| Activity | Label | Windows | Used for |
|---|---|---|---|
| Act 1–5 (5 fall types) | Fall | 629 | Fall evaluation |
| Act 6 (walking) | Active | 854 | Inactivity negative |
| Act 7 (standing) | Inactive | 844 | Inactivity positive |
| Act 8 (sitting) | Inactive | 834 | Inactivity positive |
| Act 9 (picking up) | Active | 120 | Inactivity negative |

### KTH Action Dataset
- **Source:** Schuldt et al., *ICPR* 2004 — [https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)
- **Subjects used:** 25 (all)
- **Camera:** Lateral view, 25 fps, outdoor/indoor
- **Total clips:** 200 (100 running + 100 walking)
- **Clip length:** ~15 seconds each (first 150 frames used)
- **Used for:** Running detection evaluation

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, Ultralytics YOLO, OpenCV, SciPy

---

## Evaluation

Run each detector's evaluation separately:

```bash
# Fall Detection   →  92.4%
python -m fall_detection.evaluate

# Unsafe Running   →  91.0%
python -m running_detection.evaluate

# Long-time Inactivity  →  95.8%
python -m inactivity_detection.evaluate
```

Download KTH dataset before running detection evaluation:
```bash
python -m datasets.download_running
```

---

## Real-time Demo

```bash
# Webcam
python main.py

# Video file
python main.py --source path/to/video.mp4

# RTSP stream
python main.py --source rtsp://192.168.1.10/stream
```

---

## Core Technologies

- **YOLO11s-pose** — Real-time 17-joint pose estimation
- **ByteTracker** — Multi-person persistent ID tracking
- **Butterworth Filter** — Signal smoothing for fall kinematics
- **Rule-based Logic** — No model training, fully interpretable
