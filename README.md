# Worker Abnormal Behavior Detection

A real-time rule-based system for detecting abnormal worker behaviors using pose estimation and object tracking. No model training required — pure biomechanical rules applied to YOLO11s-pose keypoints.

**[한국어](README_KO.md)**

---

## Detected Behaviors

| Behavior | Description | Accuracy |
|---|---|---|
| **Fall Detection** | Detects when a worker falls suddenly | 92.4% |
| **Unsafe Running** | Detects running in restricted/dangerous zones | 90.99% |
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
│   └── npy_loader.py             # Load pre-extracted keypoints (X.npy)
│
├── evaluation/
│   └── feature_utils.py          # Shared feature extraction helper
│
├── data/
│   ├── upfall_npy/               # Pre-extracted UP-Fall keypoints (X.npy, y.npy, meta.csv)
│   └── running_dataset/          # KTH Action Dataset clips (200 x .avi)
│
├── results/                      # Evaluation outputs saved here
├── main.py                       # Real-time demo entry point
├── requirements.txt
└── README.md / README_KO.md
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

### Train / Test Split (LOOCV)

LOOCV has no fixed split — each subject rotates through as the held-out test set while the rest are used to fit thresholds. Numbers below are per-fold averages. Fall and Inactivity are evaluated on **frame** sequences (extracted keypoint windows); Running is evaluated on whole **video clips**.

| Detector | Total frames | Train (avg/fold) | Test (avg/fold) | Folds |
|---|---|---|---|---|
| Fall | 69,150 frames (fall: 10,320 / normal: 58,830), across 131 frame sequences (59 fall + 72 normal) | ~51,863 frames | ~17,288 frames | 4 — one per UP-Fall subject |
| Inactivity | 40,500 frames (inactive: 25,530 / active: 14,970), across 48 frame sequences (24 inactive + 24 active), from 2,652 pose-estimation windows | ~30,375 frames | ~10,125 frames | 4 — one per UP-Fall subject |

| Detector | Total video clips | Frames processed | Train (avg/fold) | Test (avg/fold) | Folds |
|---|---|---|---|---|---|
| Running | 200 video clips (100 running + 100 walking) | 30,000 frames (150 frames/video clip cap; clips average 521 frames each) | ~100 video clips | ~100 video clips | 2 — odd vs even KTH subjects |

### Example Test-Data Folders

Separate from the pre-extracted LOOCV pool above, `data/` also ships **complete, single-fold test sets** built from real (not synthetic) raw data — the actual held-out data for one full LOOCV fold, not a token sample. Fall and Inactivity are frame data — raw **frame** sequences of held-out Subject1 (all 11 activities, Camera1 only); Running uses the held-out odd-subject group as whole **video clips**.

| Folder | Contents | Behavior | See this fold's result |
|---|---|---|---|
| `data/test_upfall/` | Complete Subject1 fold — **17,932 frames across 33 frame sequences** (fall activities 1–5: 2,832 frames; normal activities 6–11: 15,100 frames) | Fall | `python -m fall_detection.evaluate` — Subject1's line: **90.9%** (TN=15 FP=3 FN=0 TP=15) |
| `data/test_inactivity/` | Complete Subject1 fold — **10,431 frames across 12 frame sequences** (inactive Act7+8: 6,527 frames; active Act6+9: 3,904 frames) | Inactivity | `python -m inactivity_detection.evaluate` — Subject1's line: **100.0%** (TN=6 FP=0 FN=0 TP=6) |
| `data/test_running/` | Complete odd-subject fold — **104 video clips** (13 KTH subjects) | Running | `python main.py --source data/test_running/person01_running_d1_uncomp.avi` (visual check), or Fold 1 of `python -m running_detection.evaluate` |

> These raw folders exist for a visible, physical proof that real files sit behind the numbers — the accuracy itself is produced by the plain `evaluate` commands above (no extra flags), which is what everyone should run to reproduce the README's numbers.

> **Not committed to git.** These folders total ~10 GB (raw PNG frames are large) — far beyond what a git repo should carry, so `data/` stays in `.gitignore`. They are shared via cloud drive instead: **[Test data folders (Google Drive)](https://drive.google.com/drive/folders/1ZT5d8DihBuWDdCotHD6W1OxRpeF71I_V?usp=drive_link)** — not included in a fresh `git clone`.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, Ultralytics YOLO, OpenCV, SciPy

---

## Evaluation

`data/upfall_npy/` (pre-extracted keypoints, ~27 MB) is committed to this repo, so Fall and Inactivity run immediately after cloning. Running needs the KTH video clips, which are **not** committed (too large for git) — download them first with the public, one-command script below.

```bash
# Fall Detection   →  92.40%  (data/upfall_npy is already in the repo)
python -m fall_detection.evaluate

# Long-time Inactivity  →  95.83%  (same data/upfall_npy)
python -m inactivity_detection.evaluate

# Unsafe Running   →  90.99%
python -m datasets.download_running     # one-time: fetches data/running_dataset (~290 MB) from kth.se
python -m running_detection.evaluate    # ~30-60 min — YOLO runs on 200 video clips
```

Each script prints a full summary at the end:
```
Accuracy  : 92.40% +/- 3.4%
Precision : 87.6%
Recall    : 98.2%
F1-score  : 92.3%
[OK] >= 90%
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
