"""
Running Detection — Evaluation
Dataset : KTH Action Dataset (running vs walking, 25 subjects)
Protocol: Leave-One-Out Cross-Validation (LOOCV, odd/even split)

Command:
    python -m running_detection.evaluate
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
        label = 1 if "_running_" in fl else (0 if "_walking_" in fl else None)
        if label is None:
            continue
        clips.append({"path": os.path.join(data_dir, f),
                      "label": label, "subject": get_subject(f)})
    return clips


def extract_features(path, pose_ext, max_frames=150):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or KTH_FPS
    buf = FeatureBuffer(window_frames=30, fps=fps)
    speeds = []
    idx = 0
    while cap.isOpened() and idx < max_frames:
        ret, frame = cap.read()
        if not ret: break
        pose = pose_ext.process_frame_single(frame, idx, fps)
        buf.push(pose)
        if idx >= 29:
            speeds.append(buf.com_horizontal_speed())
        idx += 1
    cap.release()
    return {"mean_horiz_speed": np.mean(speeds) if speeds else 0.0}


def gs1(feat, y, vals):
    best_acc, best_v = 0.0, vals[0]
    for v in vals:
        acc = accuracy_score(y, (feat > v).astype(int))
        if acc > best_acc:
            best_acc, best_v = acc, v
    return best_v, best_acc


def main():
    print("=" * 55)
    print("  RUNNING DETECTION — Evaluation")
    print("  Dataset : KTH Action  |  Protocol: LOOCV")
    print("=" * 55)

    if not os.path.exists(DATA_DIR):
        print(f"\nKTH dataset not found: {DATA_DIR}")
        print("Download first: python -m datasets.download_running_dataset")
        return

    clips = list_clips(DATA_DIR)
    n_run  = sum(1 for c in clips if c["label"] == 1)
    n_walk = sum(1 for c in clips if c["label"] == 0)
    print(f"\nDataset: {len(clips)} clips  (run={n_run}  walk={n_walk})")

    print("\nExtracting features with YOLO...")
    with PoseExtractor(use_tracking=False) as pose_ext:
        import pandas as pd
        rows = []
        for clip in tqdm(clips, unit="clip"):
            f = extract_features(clip["path"], pose_ext)
            f.update({"label": clip["label"], "subject": clip["subject"]})
            rows.append(f)
    df = pd.DataFrame(rows)

    all_subjs = sorted(df["subject"].unique())
    group_a   = [s for s in all_subjs if s % 2 == 1]
    group_b   = [s for s in all_subjs if s % 2 == 0]

    accs, precs, recs, f1s = [], [], [], []
    print("\nPer-fold results:")
    for i, test_subjs in enumerate([group_a, group_b]):
        tr = df[~df["subject"].isin(test_subjs)]
        te = df[ df["subject"].isin(test_subjs)]
        thr, _ = gs1(tr["mean_horiz_speed"].values, tr["label"].values,
                     np.linspace(tr["mean_horiz_speed"].min(),
                                 tr["mean_horiz_speed"].max(), 200))
        pred = (te["mean_horiz_speed"].values > thr).astype(int)
        y    = te["label"].values
        acc  = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        cm   = confusion_matrix(y, pred, labels=[0, 1])
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"  Fold {i+1} (test={test_subjs[:3]}...): "
              f"Accuracy={acc*100:.1f}%  Precision={prec*100:.1f}%  "
              f"Recall={rec*100:.1f}%  "
              f"TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    accs = np.array(accs)
    print(f"\n{'='*55}")
    print(f"  RESULT")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accs.mean()*100:.2f}% +/- {accs.std()*100:.1f}%")
    print(f"  Precision : {np.mean(precs)*100:.1f}%")
    print(f"  Recall    : {np.mean(recs)*100:.1f}%")
    print(f"  F1-score  : {np.mean(f1s)*100:.1f}%")
    print(f"  {'[OK] >= 90%' if accs.mean() >= 0.90 else '[!!] Below 90%'}")


if __name__ == "__main__":
    main()
