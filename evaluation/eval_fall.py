"""
Fall Detection — Evaluation
Dataset : UP-Fall (pre-extracted X.npy, Subjects 1-4)
Protocol: Leave-One-Out Cross-Validation (LOOCV)

Command:
    python -m evaluation.eval_fall
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from datasets.npy_loader import load_npy_dataset
from evaluation.feature_utils import extract_window_features

ALL_SUBJECTS = [1, 2, 3, 4]
STRIDE_SEC   = 15 / 19.0


def _lowpass(sig, fc, fps, order=2):
    nyq = fps / 2.0
    b, a = butter(order, min(fc / nyq, 0.99), btype="low")
    pad  = min(3 * (max(len(a), len(b)) - 1), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=pad) if pad >= 1 else sig.copy()


def build_clips(data, feats, subjects):
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta['subject'].isin(subjects)].copy()
    meta.reset_index(drop=True, inplace=True)
    X = data['X']
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject', 'activity', 'trial']):
        idxs = grp.index.tolist()
        frames = [X[idxs[0]]]
        for i in idxs[1:]:
            frames.append(X[i][-15:])
        hip_y = np.concatenate([(f[:, 11, 1] + f[:, 12, 1]) / 2 for f in frames])
        t = np.arange(len(hip_y)) / 19.0
        try:
            hf  = _lowpass(hip_y, 4.0, 19.0)
            vel = _lowpass(np.gradient(hf, t), 8.0, 19.0)
            max_vel = float(vel.max())
        except Exception:
            max_vel = 0.0
        angles = grp['max_body_angle'].values
        ar = float(np.abs(np.diff(angles)).max() / STRIDE_SEC) if len(angles) > 1 else 0.0
        clips.append({
            'subject': subj, 'activity': act, 'trial': trial,
            'label':            int(act in {1, 2, 3, 4, 5}),
            'max_body_angle':   grp['max_body_angle'].max(),
            'min_aspect_ratio': grp['min_aspect_ratio'].min(),
            'clip_max_vel':     max_vel,
            'clip_angle_rate':  ar,
        })
    return pd.DataFrame(clips)


def predict(tr, te):
    y = tr['label'].values
    best_acc, best_pred = 0.0, np.zeros(len(te))
    for a in np.arange(40.0, 110.0, 5.0):
        for r in np.arange(10.0, 120.0, 5.0):
            acc = accuracy_score(y, ((tr['max_body_angle'] > a) & (tr['clip_angle_rate'] > r)).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_pred = ((te['max_body_angle'] > a) & (te['clip_angle_rate'] > r)).astype(int)
    return best_pred


def main():
    print("=" * 55)
    print("  FALL DETECTION — Evaluation")
    print("  Dataset : UP-Fall  |  Protocol: LOOCV")
    print("=" * 55)

    data  = load_npy_dataset()
    feats = extract_window_features(data['X'])

    clips_all = build_clips(data, feats, ALL_SUBJECTS)
    accs, precs, recs, f1s = [], [], [], []

    print("\nPer-fold results:")
    for ts in ALL_SUBJECTS:
        tr = clips_all[clips_all['subject'] != ts].reset_index(drop=True)
        te = clips_all[clips_all['subject'] == ts].reset_index(drop=True)
        pred = predict(tr, te)
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
