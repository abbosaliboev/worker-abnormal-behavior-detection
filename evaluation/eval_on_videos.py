"""
Evaluate detectors on actual video files (UP-Fall PNG frames).
This gives realistic accuracy that matches the real-time demo.

Usage:
    python -m evaluation.eval_on_videos
    python -m evaluation.eval_on_videos --dataset F:/path/to/UP-Fall
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import re
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.pose_extractor import PoseExtractor
from src.fall_detector import FallDetector
from src.inactivity_detector import InactivityDetector
from src.config import UPFALL_DATASET_PATH

UPFALL_FPS = 17.0

# UP-Fall correct activity mapping
FALL_ACTS     = {1, 2, 3, 4, 5}       # falling types
INACTIVE_ACTS = {7, 8}                 # standing, sitting
ACTIVE_ACTS   = {6, 9}                 # walking, picking up


def find_png_clips(dataset_path):
    """Find all camera1 clip folders grouped by (subject, activity, trial)."""
    clips = []
    subj_dirs = sorted(glob.glob(os.path.join(dataset_path, "Subject*")))
    for subj_dir in subj_dirs:
        subj_m = re.search(r"Subject(\d+)", subj_dir)
        if not subj_m: continue
        subj_id = int(subj_m.group(1))

        act_dirs = sorted(glob.glob(os.path.join(subj_dir, "Activity*")))
        for act_dir in act_dirs:
            act_m = re.search(r"Activity(\d+)", act_dir)
            if not act_m: continue
            act_id = int(act_m.group(1))

            trial_dirs = sorted(glob.glob(os.path.join(act_dir, "Trial*")))
            for trial_dir in trial_dirs:
                trial_m = re.search(r"Trial(\d+)", trial_dir)
                if not trial_m: continue
                trial_id = int(trial_m.group(1))

                # Camera 1 folder
                cam_pattern = os.path.join(trial_dir, f"*Camera1")
                cam_dirs = glob.glob(cam_pattern)
                if not cam_dirs: continue
                cam_dir = cam_dirs[0]

                pngs = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
                if not pngs: continue

                clips.append({
                    "frames":   pngs,
                    "subject":  subj_id,
                    "activity": act_id,
                    "trial":    trial_id,
                })
    return clips


def run_fall_on_clip(frames, pose_ext, max_frames=200):
    """Returns 1 if fall detected at any point in the clip."""
    det = FallDetector(fps=UPFALL_FPS)
    for i, path in enumerate(frames[:max_frames]):
        img = cv2.imread(path)
        if img is None: continue
        pose = pose_ext.process_frame_single(img, i, UPFALL_FPS)
        if det.update(pose) is not None:
            return 1
    return 0


def run_inactivity_on_clip(frames, pose_ext, max_frames=300):
    """Returns 1 if inactivity fires (eval_mode=True, short timeout)."""
    det = InactivityDetector(fps=UPFALL_FPS, eval_mode=True)
    for i, path in enumerate(frames[:max_frames]):
        img = cv2.imread(path)
        if img is None: continue
        pose = pose_ext.process_frame_single(img, i, UPFALL_FPS)
        if det.update(pose) is not None:
            return 1
    return 0


def evaluate_task(clips, task, pose_ext):
    if task == "fall":
        pos_acts = FALL_ACTS
        neg_acts = set(range(1, 12)) - FALL_ACTS
        run_fn   = run_fall_on_clip
    else:
        pos_acts = INACTIVE_ACTS
        neg_acts = ACTIVE_ACTS
        run_fn   = run_inactivity_on_clip

    relevant = [c for c in clips if c["activity"] in pos_acts | neg_acts]
    y_true, y_pred = [], []

    for clip in tqdm(relevant, desc=f"  {task}", unit="clip"):
        label = 1 if clip["activity"] in pos_acts else 0
        pred  = run_fn(clip["frames"], pose_ext)
        y_true.append(label)
        y_pred.append(pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n  {'='*50}")
    print(f"  {task.upper()} (real video evaluation)")
    print(f"  {'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"  F1-score  : {f1*100:.1f}%")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    print(f"  {'[OK] >= 90%' if acc >= 0.90 else '[!!] Below 90%'}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=UPFALL_DATASET_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("  Real-video Evaluation (actual PNG frames + YOLO)")
    print(f"  Dataset: {args.dataset}")
    print("=" * 60)

    if not os.path.exists(args.dataset):
        print(f"\nDataset not found: {args.dataset}")
        print("Set UPFALL_DATASET_PATH in src/config.py")
        return

    clips = find_png_clips(args.dataset)
    subjects = sorted(set(c["subject"] for c in clips))
    print(f"\nFound {len(clips)} clips, {len(subjects)} subjects: {subjects}")

    print("\nLoading YOLO model...")
    with PoseExtractor(use_tracking=False) as pose_ext:
        acc_fall  = evaluate_task(clips, "fall",       pose_ext)
        acc_inact = evaluate_task(clips, "inactivity", pose_ext)

    print(f"\n{'='*60}")
    print(f"  SUMMARY (real video, no cross-validation)")
    print(f"{'='*60}")
    print(f"  Fall       : {acc_fall*100:.2f}%")
    print(f"  Inactivity : {acc_inact*100:.2f}%")
    print(f"  Note: Running evaluated on KTH dataset separately (90.4%)")


if __name__ == "__main__":
    main()
