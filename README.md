# Worker Abnormal Behavior Detection

A real-time rule-based system for detecting abnormal worker behaviors using pose estimation and object tracking. No model training required — pure biomechanical rules applied to YOLO11n-pose keypoints.

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
│   ├── pose_extractor.py         # YOLO11n-pose + ByteTracker
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
YOLO11n-pose  →  17 body keypoints per person
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
- Computes **body tilt angle** and **angular rate of change** (°/sec)
- Rule: `body_angle > 70° AND angle_rate > 65°/sec`
- Key insight: falls are rapid (74–140°/sec), deliberate lying-down is slow (2–5°/sec)

### Unsafe Running Logic
- Tracks **horizontal center-of-mass speed** frame by frame
- Rule: `horizontal_speed > calibrated_threshold`
- Running is ~2× faster than walking in lateral-camera setups

### Inactivity Logic
- Measures **fraction of still frames** and **posture stability**
- Rule: `still_fraction > 0.70 AND body_angle_std < 3.5°`
- Timer: alert fires after 5 continuous minutes of stillness

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
- Martinez-Velasco et al., *Data* 2019
- Activities: falling (5 types), walking, standing, sitting, picking up
- Used for: Fall and Inactivity evaluation

### KTH Action Dataset
- Schuldt et al., *ICPR* 2004
- 25 subjects, 200 clips (100 running + 100 walking)
- Used for: Running detection evaluation

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

# Unsafe Running   →  90.4%
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

- **YOLO11n-pose** — Real-time 17-joint pose estimation
- **ByteTracker** — Multi-person persistent ID tracking
- **Butterworth Filter** — Signal smoothing for fall kinematics
- **Rule-based Logic** — No model training, fully interpretable
