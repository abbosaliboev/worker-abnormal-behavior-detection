# Worker Abnormal Behavior Detection

A real-time rule-based system for detecting abnormal worker behaviors using pose estimation and object tracking. No model training required — pure biomechanical rules applied to YOLO11n-pose keypoints.

**[한국어](README_KO.md) | [O'zbek](README_UZ.md)**

---

## Detected Behaviors

| Behavior | Description | Accuracy |
|---|---|---|
| **Fall Detection** | Detects when a worker falls suddenly | 92.4% |
| **Unsafe Running** | Detects running in restricted/dangerous zones | 90.4% |
| **Long-time Inactivity** | Detects workers motionless for 5+ minutes | 95.8% |

> Evaluated using Leave-One-Out Cross-Validation (LOOCV) across subjects.

---

## How It Works

```
CCTV / Camera
      ↓
YOLO11n-pose  →  17 body keypoints per person
      ↓
ByteTracker   →  Unique ID assigned to each worker
      ↓
┌──────────────┬─────────────────┬───────────────────┐
│ Fall         │ Running         │ Inactivity        │
│ Detector     │ Detector        │ Detector          │
└──────────────┴─────────────────┴───────────────────┘
      ↓
Alert  (FALL | RUNNING | INACTIVITY)
```

### Fall Detection Logic
- Computes **body tilt angle** (degrees from vertical) and **angular rate of change** (°/sec)
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
| Running | **90.42%** ± 1.9% | KTH Action (25 subjects) | LOOCV |
| Inactivity | **95.83%** ± 4.2% | UP-Fall (4 subjects) | LOOCV |
| **Average** | **92.88%** | | |

---

## Datasets

### UP-Fall Detection Dataset
- Martinez-Velasco et al., *Data* 2019
- Activities: falling (4 types), walking, standing, sitting, picking up
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

## Usage

### Real-time Demo
```bash
# Webcam
python main.py

# Video file
python main.py --source path/to/video.mp4

# RTSP stream
python main.py --source rtsp://192.168.1.10/stream

# Disable tracking (single person)
python main.py --no-tracking
```

### Evaluation
```bash
# Fall + Inactivity (UP-Fall dataset)
python -m evaluation.evaluate

# Running detection (KTH dataset)
python -m evaluation.eval_running_kth_calibrated

# Full inactivity evaluation
python -m evaluation.eval_inactivity_full
```

### Download KTH Dataset
```bash
python -m datasets.download_running_dataset
```

---

## Project Structure

```
├── src/
│   ├── config.py               # Thresholds and settings
│   ├── pose_extractor.py       # YOLO11n-pose + ByteTracker
│   ├── feature_extractor.py    # Biomechanical features
│   ├── fall_detector.py        # Fall detection rules
│   ├── running_detector.py     # Running detection rules
│   ├── inactivity_detector.py  # Per-person inactivity timer
│   └── behavior_monitor.py     # Orchestrates all detectors
├── evaluation/
│   ├── evaluate.py                      # Fall + Inactivity LOOCV
│   ├── eval_running_kth_calibrated.py   # Running LOOCV (KTH)
│   └── eval_inactivity_full.py          # Full inactivity evaluation
├── datasets/
│   ├── npy_loader.py                    # Load pre-extracted X.npy
│   └── download_running_dataset.py      # KTH dataset downloader
├── main.py                              # Real-time demo
├── requirements.txt
└── REPORT.md                            # Detailed technical report
```

---

## Core Technologies

- **YOLO11n-pose** — Real-time 17-joint pose estimation
- **ByteTracker** — Multi-person persistent ID tracking
- **Butterworth Filter** — Signal smoothing for fall kinematics
- **Rule-based Logic** — No model training, fully interpretable

---

## Key Design Decisions

- **No training required** — Rules are derived from biomechanics and calibrated on dataset statistics
- **Multi-person** — ByteTracker gives each worker a unique ID with independent timers
- **Cascade architecture** — Fall → Running → Inactivity (each detector runs only if the previous didn't fire)
- **Threshold calibration** — Per-fold calibration in LOOCV ensures cross-subject generalization
