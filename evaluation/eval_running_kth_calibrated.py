"""
KTH Running evaluation with per-fold threshold calibration.

Step 1: Extract clip-level features from ALL clips (YOLO + FeatureBuffer)
Step 2: Calibrated LOOCV — find optimal thresholds on training split
Step 3: Apply to test split

Features:
  - mean_horiz_speed  : mean CoM horizontal displacement per frame
  - mean_step_freq    : mean step frequency (Hz)
  - mean_vert_osc     : mean vertical CoM oscillation
  - mean_knee_var     : mean knee angle variance

In KTH: running moves faster laterally than walking
  → horiz_speed and step_freq are the key discriminators
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.pose_extractor import PoseExtractor
from src.feature_extractor import FeatureBuffer
from src.config import YOLO_POSE_MODEL

DATA_DIR = r"f:\Project_F\Company_Abnormal_Project\data\running_dataset"
KTH_FPS  = 25.0


def get_subject(fname):
    m = re.search(r"person(\d+)", fname, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def list_clips(data_dir):
    clips = []
    for f in sorted(os.listdir(data_dir)):
        if not f.lower().endswith(".avi"):
            continue
        fl = f.lower()
        if "_running_" in fl:
            label = 1
        elif "_walking_" in fl:
            label = 0
        else:
            continue
        clips.append({"path": os.path.join(data_dir, f),
                      "label": label, "subject": get_subject(f)})
    return clips


def extract_clip_features(video_path, pose_ext, max_frames=150):
    """
    Run YOLO on clip frames and extract aggregate FeatureBuffer statistics.
    Returns dict with clip-level feature means.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or KTH_FPS
    buf = FeatureBuffer(window_frames=30, fps=fps)
    idx = 0
    h_speeds, step_freqs, v_oscs, k_vars = [], [], [], []

    while cap.isOpened() and idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        pose = pose_ext.process_frame_single(frame, idx, fps)
        buf.push(pose)
        if idx >= 29:   # buffer full
            h_speeds.append(buf.com_horizontal_speed())
            step_freqs.append(buf.step_frequency())
            v_oscs.append(buf.com_vertical_oscillation())
            k_vars.append(buf.knee_angle_variance())
        idx += 1

    cap.release()
    if not h_speeds:
        return None
    return {
        "mean_horiz_speed": np.mean(h_speeds),
        "mean_step_freq":   np.mean(step_freqs),
        "mean_vert_osc":    np.mean(v_oscs),
        "mean_knee_var":    np.mean(k_vars),
    }


def gs1(feat, y, vals, direction="above"):
    best_acc, best_v = 0.0, vals[0]
    for v in vals:
        p = (feat > v) if direction == "above" else (feat < v)
        acc = accuracy_score(y, p.astype(int))
        if acc > best_acc:
            best_acc, best_v = acc, v
    return best_v, best_acc


def main():
    print("=" * 60)
    print("  Running Detection — KTH (Calibrated LOOCV)")
    print("=" * 60)

    clips = list_clips(DATA_DIR)
    n_run  = sum(1 for c in clips if c["label"] == 1)
    n_walk = sum(1 for c in clips if c["label"] == 0)
    print(f"\nDataset: {len(clips)} clips  (run={n_run}  walk={n_walk})")

    print("\nExtracting clip features (YOLO, first 150 frames per clip)...")
    with PoseExtractor() as pose_ext:
        features = []
        for clip in tqdm(clips, unit="clip"):
            f = extract_clip_features(clip["path"], pose_ext)
            if f is not None:
                f.update({"label": clip["label"], "subject": clip["subject"]})
                features.append(f)

    print(f"  Extracted features for {len(features)} clips")

    # Feature stats
    import pandas as pd
    df = pd.DataFrame(features)
    print("\nFeature distributions:")
    for feat in ["mean_horiz_speed", "mean_step_freq", "mean_vert_osc", "mean_knee_var"]:
        pos = df[df["label"]==1][feat]
        neg = df[df["label"]==0][feat]
        print(f"  {feat}:")
        print(f"    RUN : mean={pos.mean():.5f}  p50={pos.median():.5f}")
        print(f"    WALK: mean={neg.mean():.5f}  p50={neg.median():.5f}")

    # LOOCV with calibration
    all_subjs = sorted(df["subject"].unique())
    group_a = [s for s in all_subjs if s % 2 == 1]
    group_b = [s for s in all_subjs if s % 2 == 0]

    print("\nLOOCV (odd/even subject split):")
    accs, precs, recs, f1s = [], [], [], []

    for i, test_subjs in enumerate([group_a, group_b]):
        tr = df[~df["subject"].isin(test_subjs)]
        te = df[ df["subject"].isin(test_subjs)]
        y_tr = tr["label"].values
        y_te = te["label"].values

        # Grid-search best single feature threshold on training
        best_acc_tr, best_thr, best_feat = 0.0, 0.0, "mean_horiz_speed"
        for feat in ["mean_horiz_speed", "mean_step_freq", "mean_vert_osc"]:
            vals = np.linspace(tr[feat].min(), tr[feat].max(), 200)
            v, a = gs1(tr[feat].values, y_tr, vals, direction="above")
            if a > best_acc_tr:
                best_acc_tr, best_thr, best_feat = a, v, feat

        pred = (te[best_feat].values > best_thr).astype(int)
        acc  = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score(y_te, pred, zero_division=0)
        f1   = f1_score(y_te, pred, zero_division=0)
        cm   = confusion_matrix(y_te, pred, labels=[0, 1])
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

        print(f"  Fold {i+1}: acc={acc*100:.1f}%  prec={prec*100:.1f}%  "
              f"rec={rec*100:.1f}%  f1={f1*100:.1f}%"
              f"  ({best_feat}>{best_thr:.4f}  train_acc={best_acc_tr*100:.1f}%)"
              f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    accs = np.array(accs)
    print(f"\n{'='*60}")
    print(f"  RUNNING — KTH (calibrated, first 150 frames/clip)")
    print(f"{'='*60}")
    print(f"  Accuracy  : {accs.mean()*100:.2f}% +/- {accs.std()*100:.1f}%")
    print(f"  Precision : {np.mean(precs)*100:.1f}%")
    print(f"  Recall    : {np.mean(recs)*100:.1f}%")
    print(f"  F1-score  : {np.mean(f1s)*100:.1f}%")
    print(f"  {'[OK] >= 90%' if accs.mean() >= 0.90 else '[!!] Below 90%'}")


if __name__ == "__main__":
    main()
