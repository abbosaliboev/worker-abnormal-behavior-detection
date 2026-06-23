"""
Inactivity Detection — Evaluation
Dataset : UP-Fall (pre-extracted X.npy, Subjects 1-4)
Protocol: Leave-One-Out Cross-Validation (LOOCV)

Positive: Act7 (standing) + Act8 (sitting)
Negative: Act6 (walking)  + Act9 (picking up)

Command:
    python -m inactivity_detection.evaluate
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from datasets.npy_loader import load_npy_dataset
from evaluation.feature_utils import extract_window_features

ALL_SUBJECTS = [1, 2, 3, 4]
INACT_POS    = {7, 8}    # standing, sitting
INACT_NEG    = {6, 9}    # walking, picking up


def build_clips(data, feats, subjects):
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    X = data['X']
    com_y = ((X[:,:,5,1]+X[:,:,6,1]+X[:,:,11,1]+X[:,:,12,1])/4).mean(axis=1)
    meta['com_y'] = com_y
    mask = meta['subject'].isin(subjects) & meta['activity'].isin(INACT_POS | INACT_NEG)
    meta = meta[mask].copy().reset_index(drop=True)
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject', 'activity', 'trial']):
        still_fraction = (grp['mean_kp_disp'] < 0.005).mean()
        clips.append({
            'subject':        subj,
            'activity':       act,
            'trial':          trial,
            'label':          int(act in INACT_POS),
            'still_fraction': still_fraction,
            'body_angle_std': grp['max_body_angle'].std() if len(grp) > 1 else 0.0,
        })
    return pd.DataFrame(clips)


def main():
    print("=" * 55)
    print("  INACTIVITY DETECTION — Evaluation")
    print("  Dataset : UP-Fall  |  Protocol: LOOCV")
    print("  Positive: standing (Act7) + sitting (Act8)")
    print("  Negative: walking  (Act6) + picking (Act9)")
    print("=" * 55)

    data  = load_npy_dataset()
    feats = extract_window_features(data['X'])
    clips_all = build_clips(data, feats, ALL_SUBJECTS)

    accs, precs, recs, f1s = [], [], [], []
    print("\nPer-fold results:")
    for ts in ALL_SUBJECTS:
        tr = clips_all[clips_all['subject'] != ts].reset_index(drop=True)
        te = clips_all[clips_all['subject'] == ts].reset_index(drop=True)

        # Midpoint threshold (robust across subjects)
        t_sf  = (tr[tr['label']==1]['still_fraction'].mean() +
                 tr[tr['label']==0]['still_fraction'].mean()) / 2.0
        # Conservative angle threshold: inactive_max + buffer
        t_ang = tr[tr['label']==1]['body_angle_std'].max() + 0.5

        pred = ((te['still_fraction']  > t_sf) &
                (te['body_angle_std'] < t_ang)).astype(int)
        y    = te['label'].values
        acc  = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        cm   = confusion_matrix(y, pred, labels=[0, 1])
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"  Subject {ts}: Accuracy={acc*100:.1f}%  "
              f"Precision={prec*100:.1f}%  Recall={rec*100:.1f}%  "
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
