"""
Running detector evaluation on KTH Action Dataset.

Pipeline:
  1. Load Running and Walking videos from KTH dataset
  2. Extract YOLO11n-pose keypoints from each video
  3. Feed frame-by-frame to the rule-based RunningDetector
  4. Evaluate: did the detector fire for running clips? Not fire for walking?

LOOCV by subject (KTH has 25 subjects labeled in filename):
  e.g. "person01_running_d1_uncomp.avi" -> subject 01

Usage:
    python -m evaluation.eval_running_kth
    python -m evaluation.eval_running_kth --data_dir data/running_dataset
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

from src.pose_extractor import PoseExtractor
from src.running_detector import RunningDetector
from src.config import STGCN_FPS

DEFAULT_DATA_DIR = r"f:\Project_F\Company_Abnormal_Project\data\running_dataset"
KTH_FPS = 25.0


# ─── helpers ─────────────────────────────────────────────────────────────────

def get_subject_id(filename: str) -> int:
    """Extract subject number from KTH filename: person01_running_d1.avi -> 1"""
    m = re.search(r"person(\d+)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def list_clips(data_dir: str) -> list[dict]:
    """
    Return list of {path, label(0=walk,1=run), subject} dicts.
    Handles two layouts:
      a) data_dir/Running/*.avi  +  data_dir/Walking/*.avi  (subfolder layout)
      b) data_dir/person*_running_*.avi  (KTH flat layout — all in root)
    """
    clips = []

    # Try subfolder layout first
    for cls, label in [("Running", 1), ("Walking", 0)]:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith((".avi", ".mp4", ".mov")):
                    clips.append({
                        "path":    os.path.join(cls_dir, f),
                        "label":   label,
                        "subject": get_subject_id(f),
                        "name":    f,
                    })

    if clips:
        return clips

    # Flat layout (KTH extracts all files to root)
    for f in sorted(os.listdir(data_dir)):
        if not f.lower().endswith((".avi", ".mp4", ".mov")):
            continue
        fl = f.lower()
        if "_running_" in fl:
            label = 1
        elif "_walking_" in fl:
            label = 0
        else:
            continue
        clips.append({
            "path":    os.path.join(data_dir, f),
            "label":   label,
            "subject": get_subject_id(f),
            "name":    f,
        })

    return clips


def predict_clip(video_path: str, pose_ext: PoseExtractor) -> int:
    """
    Returns 1 if RunningDetector fires at any point in the clip, else 0.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or KTH_FPS
    det = RunningDetector(fps=fps)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pose = pose_ext.process_frame_single(frame, frame_idx, fps)
        if det.update(pose) is not None:
            cap.release()
            return 1
        frame_idx += 1

    cap.release()
    return 0


# ─── evaluation ──────────────────────────────────────────────────────────────

def evaluate(clips: list[dict], pose_ext: PoseExtractor) -> dict:
    y_true, y_pred, subjects = [], [], []
    for clip in tqdm(clips, desc="  Evaluating", unit="clip"):
        pred = predict_clip(clip["path"], pose_ext)
        y_true.append(clip["label"])
        y_pred.append(pred)
        subjects.append(clip["subject"])
    return {"y_true": y_true, "y_pred": y_pred, "subjects": subjects}


def loocv(clips: list[dict], pose_ext: PoseExtractor):
    """Leave-one-subject-out cross-validation."""
    all_subjs = sorted(set(c["subject"] for c in clips))
    # Use groups of subjects to avoid single-subject test (too small)
    # Split: odd subjects = group A, even = group B
    group_a = [s for s in all_subjs if s % 2 == 1]
    group_b = [s for s in all_subjs if s % 2 == 0]
    folds   = [group_a, group_b]

    accs, precs, recs, f1s = [], [], [], []
    for i, test_subjs in enumerate(folds):
        train_clips = [c for c in clips if c["subject"] not in test_subjs]
        test_clips  = [c for c in clips if c["subject"] in test_subjs]

        if not test_clips:
            continue

        print(f"\n  Fold {i+1}: test subjects {test_subjs[:3]}...  "
              f"({len(test_clips)} test clips)")

        result = evaluate(test_clips, pose_ext)
        y, p = result["y_true"], result["y_pred"]

        if sum(y) == 0 or sum(y) == len(y):
            print("  Skipped: degenerate split")
            continue

        acc  = accuracy_score(y, p)
        prec = precision_score(y, p, zero_division=0)
        rec  = recall_score(y, p, zero_division=0)
        f1   = f1_score(y, p, zero_division=0)
        cm   = confusion_matrix(y, p, labels=[0, 1])
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

        print(f"  acc={acc*100:.1f}%  prec={prec*100:.1f}%  "
              f"rec={rec*100:.1f}%  f1={f1*100:.1f}%")
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    return np.array(accs), np.array(precs), np.array(recs), np.array(f1s)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("  Running Detector Evaluation — KTH Action Dataset")
    print("=" * 60)

    clips = list_clips(args.data_dir)
    if not clips:
        print("\nNo video clips found. Run first:")
        print("  python -m datasets.download_running_dataset")
        return

    n_run  = sum(1 for c in clips if c["label"] == 1)
    n_walk = sum(1 for c in clips if c["label"] == 0)
    n_subj = len(set(c["subject"] for c in clips))
    print(f"\nDataset: {len(clips)} clips  "
          f"(running={n_run}  walking={n_walk}  subjects={n_subj})")

    print("\nLoading YOLO11n-pose model...")
    with PoseExtractor() as pose_ext:
        accs, precs, recs, f1s = loocv(clips, pose_ext)

    if len(accs) == 0:
        print("No valid folds found.")
        return

    print("\n" + "=" * 60)
    print("  RUNNING DETECTION — KTH Dataset Results")
    print("=" * 60)
    print(f"  Accuracy  : {accs.mean()*100:.2f}% +/- {accs.std()*100:.1f}%")
    print(f"  Precision : {precs.mean()*100:.1f}%")
    print(f"  Recall    : {recs.mean()*100:.1f}%")
    print(f"  F1-score  : {f1s.mean()*100:.1f}%")
    print(f"  {'[OK] >= 90%' if accs.mean() >= 0.90 else '[!!] Below 90%'}")

    print("\nNote: This evaluates the FULL running detector pipeline")
    print("  (YOLO pose -> feature extraction -> rule-based classification)")
    print("  KTH has lateral camera -> high horizontal speed visible")


if __name__ == "__main__":
    main()
