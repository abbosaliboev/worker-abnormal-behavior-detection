"""
Master evaluation script — Worker Abnormal Behavior Detection.

Runs LOOCV (4-fold, leave-one-subject-out) for all three detectors.

Usage:
    python -m evaluation.evaluate

Results (UP-Fall dataset, Subjects 1-4):
    Fall       : 92.40% +/- 3.4%
    Running    : 93.75% +/- 10.8%
    Inactivity : 100.0% +/- 0.0%
    Mean       : 95.38%
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from datasets.npy_loader import load_npy_dataset
from evaluation.eval_from_npy import extract_window_features

ALL_SUBJECTS = [1, 2, 3, 4]
STGCN_FPS    = 19.0
STRIDE_SEC   = 15 / STGCN_FPS   # ~0.79 sec between windows


# ─── helpers ─────────────────────────────────────────────────────────────────

def _lowpass(sig, fc, fps, order=2):
    nyq = fps / 2.0
    wn  = min(fc / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    pad  = min(3 * (max(len(a), len(b)) - 1), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=pad) if pad >= 1 else sig.copy()


def report(name, fold_accs, fold_precs, fold_recs, fold_f1s):
    accs = np.array(fold_accs)
    ok   = "[OK] >= 90%" if accs.mean() >= 0.90 else "[!!] Below 90%"
    print(f"\n{'='*52}")
    print(f"  {name.upper()}")
    print(f"{'='*52}")
    print(f"  Accuracy  : {accs.mean()*100:6.2f}% +/- {accs.std()*100:.1f}%")
    print(f"  Precision : {np.mean(fold_precs)*100:.1f}%")
    print(f"  Recall    : {np.mean(fold_recs)*100:.1f}%")
    print(f"  F1-score  : {np.mean(fold_f1s)*100:.1f}%")
    print(f"  {ok}")
    return accs.mean()


# ─── Feature extraction for clips ────────────────────────────────────────────

def build_fall_clips(data, feats, subjects):
    """Fall: Act 1-5 (positive) vs Act 6-11 (negative). Clip-level."""
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta['subject'].isin(subjects)].copy()
    meta.reset_index(drop=True, inplace=True)
    X = data['X']
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject','activity','trial']):
        idxs = grp.index.tolist()
        frames_list = [X[idxs[0]]]
        for i in idxs[1:]:
            frames_list.append(X[i][-15:])
        hip_y_seq = np.concatenate(
            [(f[:, 11, 1] + f[:, 12, 1]) / 2 for f in frames_list]
        )
        t = np.arange(len(hip_y_seq)) / STGCN_FPS
        if len(hip_y_seq) >= 6:
            try:
                hf  = _lowpass(hip_y_seq, 4.0, STGCN_FPS)
                vel = _lowpass(np.gradient(hf, t), 8.0, STGCN_FPS)
                max_vel = float(vel.max())
            except Exception:
                max_vel = 0.0
        else:
            max_vel = 0.0
        # angle rate: max |angle_{i+1} - angle_i| / stride_sec
        angles = grp['max_body_angle'].values
        ar     = float(np.abs(np.diff(angles)).max() / STRIDE_SEC) if len(angles) > 1 else 0.0
        clips.append({
            'subject': subj, 'activity': act, 'trial': trial,
            'label':            int(act in {1,2,3,4,5}),
            'max_body_angle':   grp['max_body_angle'].max(),
            'min_aspect_ratio': grp['min_aspect_ratio'].min(),
            'clip_max_vel':     max_vel,
            'clip_angle_rate':  ar,   # deg/sec
        })
    return pd.DataFrame(clips)


def build_run_clips(data, feats, subjects):
    """Running: Act 8 (positive) vs Act 6,9,11 (negative). Clip-level."""
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta['subject'].isin(subjects) &
                meta['activity'].isin({6,8,9,11})].copy()
    meta.reset_index(drop=True, inplace=True)
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject','activity','trial']):
        clips.append({
            'subject': subj, 'activity': act, 'trial': trial,
            'label':          int(act == 8),
            'mean_kp_disp':   grp['mean_kp_disp'].mean(),
        })
    return pd.DataFrame(clips)


def build_inact_clips(data, feats, subjects):
    """Inactivity: Act 7 (positive) vs Act 6,9,11 (negative). Clip-level tracking."""
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    X = data['X']
    # CoM per window
    xy = X[:, :, :, :2]
    com_y_w = ((xy[:,:,5,1]+xy[:,:,6,1]+xy[:,:,11,1]+xy[:,:,12,1])/4).mean(axis=1)
    meta['com_y'] = com_y_w

    meta = meta[meta['subject'].isin(subjects) &
                meta['activity'].isin({6,7,9,11})].copy()
    meta.reset_index(drop=True, inplace=True)
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject','activity','trial']):
        clips.append({
            'subject': subj, 'activity': act, 'trial': trial,
            'label':            int(act == 7),
            'mean_kp_disp':     grp['mean_kp_disp'].mean(),
            'com_y_std':        grp['com_y'].std() if len(grp) > 1 else 0.0,
            'body_angle_std':   grp['max_body_angle'].std() if len(grp) > 1 else 0.0,
        })
    return pd.DataFrame(clips)


# ─── Per-fold predictors ──────────────────────────────────────────────────────

def predict_fall(tr, te):
    y = tr['label'].values
    best_acc, best_pred = 0.0, np.zeros(len(te))
    for a in np.arange(40.0, 110.0, 5.0):
        for r in np.arange(10.0, 120.0, 5.0):
            c1 = tr['max_body_angle']  > a
            c2 = tr['clip_angle_rate'] > r
            acc = accuracy_score(y, (c1 & c2).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_pred = ((te['max_body_angle']  > a) &
                             (te['clip_angle_rate'] > r)).astype(int)
    return best_pred


def predict_running(tr, te):
    y = tr['label'].values
    best_acc, best_thr = 0.0, 0.005
    for v in np.arange(0.0005, 0.015, 0.0005):
        acc = accuracy_score(y, (tr['mean_kp_disp'] < v).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, v
    return (te['mean_kp_disp'] < best_thr).astype(int)


def predict_inactivity(tr, te):
    """AND rule with midpoint thresholds per feature."""
    def midpt(feat):
        pos = tr[tr['label'] == 1][feat].mean()
        neg = tr[tr['label'] == 0][feat].mean()
        return (pos + neg) / 2.0
    t1 = midpt('mean_kp_disp')
    t2 = midpt('com_y_std')
    t3 = midpt('body_angle_std')
    return ((te['mean_kp_disp']   < t1) &
            (te['com_y_std']      < t2) &
            (te['body_angle_std'] < t3)).astype(int)


# ─── LOOCV runner ─────────────────────────────────────────────────────────────

def loocv(clips_all, predict_fn):
    accs, precs, recs, f1s = [], [], [], []
    for test_subj in ALL_SUBJECTS:
        tr = clips_all[clips_all['subject'] != test_subj].reset_index(drop=True)
        te = clips_all[clips_all['subject'] == test_subj].reset_index(drop=True)
        if len(te) == 0 or te['label'].sum() == 0 or te['label'].sum() == len(te):
            continue
        pred = predict_fn(tr, te)
        accs.append(accuracy_score(te['label'], pred))
        precs.append(precision_score(te['label'], pred, zero_division=0))
        recs.append(recall_score(te['label'], pred, zero_division=0))
        f1s.append(f1_score(te['label'], pred, zero_division=0))
        print(f"    S{test_subj}: acc={accs[-1]*100:.1f}%  "
              f"prec={precs[-1]*100:.1f}%  rec={recs[-1]*100:.1f}%")
    return accs, precs, recs, f1s


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Worker Abnormal Behavior Detection — Evaluation")
    print("  Method: Rule-based (YOLO11n-pose keypoints)")
    print("  Protocol: LOOCV across 4 subjects (UP-Fall dataset)")
    print("=" * 60)

    data  = load_npy_dataset()
    feats = extract_window_features(data['X'])

    print("\nBuilding clip-level features...")
    fall_clips  = build_fall_clips(data, feats, ALL_SUBJECTS)
    run_clips   = build_run_clips(data, feats, ALL_SUBJECTS)
    inact_clips = build_inact_clips(data, feats, ALL_SUBJECTS)

    results = {}

    print("\n[1/3] Fall Detection  (Act1-5 positive | Act6-11 negative)")
    a,p,r,f = loocv(fall_clips, predict_fall)
    results["Fall"] = report("Fall", a, p, r, f)

    print("\n[2/3] Running Detection  (Act8 positive | Act6,9,11 negative)")
    a,p,r,f = loocv(run_clips, predict_running)
    results["Running"] = report("Running", a, p, r, f)

    print("\n[3/3] Inactivity Detection  (Act7 positive | Act6,9,11 negative)")
    print("      Features: kp_disp + com_y_std + body_angle_std  (AND rule)")
    a,p,r,f = loocv(inact_clips, predict_inactivity)
    results["Inactivity"] = report("Inactivity", a, p, r, f)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<14} {'Accuracy':>10}  {'Status'}")
    print(f"  {'-'*38}")
    for name, acc in results.items():
        ok = "[OK] >= 90%" if acc >= 0.90 else "[!!] Below 90%"
        print(f"  {name:<14} {acc*100:>9.2f}%  {ok}")
    mean = np.mean(list(results.values()))
    print(f"\n  Mean accuracy : {mean*100:.2f}%")
    if all(v >= 0.90 for v in results.values()):
        print("  All tasks meet >= 90% target.")


if __name__ == "__main__":
    main()
