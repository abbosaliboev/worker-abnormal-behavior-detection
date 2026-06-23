"""
Final evaluation targeting >= 90% for all three detectors.

Key improvements:
  FALL       : Add clip_max_angle_rate (angle change speed between windows)
               Falls are rapid (70-80 deg/sec), lying-down is slow (15-25 deg/sec)
  RUNNING    : Keep mean_kp_disp < thr  (93.75% already)
  INACTIVITY : Use class-midpoint threshold instead of minimum threshold
               Robust to cross-subject variability in standing motion level
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


STGCN_FPS   = 19.0
STGCN_STRIDE = 15   # frames between windows
STRIDE_SEC  = STGCN_STRIDE / STGCN_FPS   # ~0.79 seconds per stride


# ─── helpers ─────────────────────────────────────────────────────────────────

def report(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n  Accuracy : {acc*100:6.2f}%  Precision: {prec*100:.1f}%"
          f"  Recall: {rec*100:.1f}%  F1: {f1*100:.1f}%")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}"
          f"  {'[OK] >= 90%' if acc >= 0.90 else '[!!] Below 90%'}")
    return acc


def _lowpass(sig, fc, fps, order=2):
    nyq = fps / 2.0
    wn  = min(fc / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    pad  = min(3 * (max(len(a), len(b)) - 1), len(sig) - 1)
    if pad < 1:
        return sig.copy()
    return filtfilt(b, a, sig, padlen=pad)


# ─── Build clips ──────────────────────────────────────────────────────────────

def build_fall_clips(data, feats, subjects, positive_acts, mask_acts):
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta['subject'].isin(subjects) & meta['activity'].isin(mask_acts)].copy()
    meta.reset_index(drop=True, inplace=True)
    X = data['X']
    clips = []

    for (subj, act, trial), grp in meta.groupby(['subject', 'activity', 'trial']):
        idxs = grp.index.tolist()

        # Reconstruct full clip sequence (non-overlapping)
        frames = [X[idxs[0]]]
        for i in idxs[1:]:
            frames.append(X[i][-15:])
        clip_frames = np.concatenate(frames, axis=0)  # (T_total, 17, 3)

        # Butterworth-filtered hip velocity
        hip_y = (clip_frames[:, 11, 1] + clip_frames[:, 12, 1]) / 2.0
        t = np.arange(len(hip_y)) / STGCN_FPS
        if len(hip_y) >= 6:
            try:
                hip_f = _lowpass(hip_y, 4.0, STGCN_FPS)
                vel   = np.gradient(hip_f, t)
                vel_f = _lowpass(vel, 8.0, STGCN_FPS)
                clip_max_vel = float(vel_f.max())
            except Exception:
                clip_max_vel = float(np.gradient(hip_y, t).max())
        else:
            clip_max_vel = 0.0

        # Max angle rate of change between consecutive windows
        # Use per-window max_body_angle values
        angles = grp['max_body_angle'].values   # one per window
        if len(angles) >= 2:
            angle_deltas = np.abs(np.diff(angles))   # deg per stride
            clip_max_angle_rate = float(angle_deltas.max() / STRIDE_SEC)  # deg/sec
        else:
            clip_max_angle_rate = 0.0

        clips.append({
            'subject':              subj,
            'activity':             act,
            'trial':                trial,
            'label':                int(act in positive_acts),
            'max_body_angle':       grp['max_body_angle'].max(),
            'min_aspect_ratio':     grp['min_aspect_ratio'].min(),
            'max_hip_y':            grp['max_hip_y'].max(),
            'mean_kp_disp':         grp['mean_kp_disp'].mean(),
            'clip_max_vel':         clip_max_vel,
            'clip_max_angle_rate':  clip_max_angle_rate,   # deg/sec
        })

    return pd.DataFrame(clips)


def build_clips_simple(data, feats, subjects, positive_acts, mask_acts):
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta['subject'].isin(subjects) & meta['activity'].isin(mask_acts)].copy()
    meta.reset_index(drop=True, inplace=True)
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject', 'activity', 'trial']):
        clips.append({
            'subject':      subj, 'activity': act, 'trial': trial,
            'label':        int(act in positive_acts),
            'mean_kp_disp': grp['mean_kp_disp'].mean(),
        })
    return pd.DataFrame(clips)


# ─── Threshold selection ──────────────────────────────────────────────────────

def gs1(feat, y, vals, direction="above"):
    best_acc, best_v = 0.0, vals[0]
    for v in vals:
        p = (feat > v) if direction == "above" else (feat < v)
        acc = accuracy_score(y, p.astype(int))
        if acc > best_acc:
            best_acc, best_v = acc, v
    return best_v, best_acc


def midpoint_threshold(feat, y, direction="below"):
    """Use midpoint between class centroids — more robust cross-subject."""
    pos_mean = feat[y == 1].mean()
    neg_mean = feat[y == 0].mean()
    return float((pos_mean + neg_mean) / 2.0)


def gs3_and(f1, f2, f3, y, v1s, v2s, v3s, d1="above", d2="above", d3="above"):
    """Grid search for 3-condition AND combination."""
    best_acc, best = 0.0, (v1s[0], v2s[0], v3s[0])
    for v1 in v1s:
        c1 = (f1 > v1) if d1 == "above" else (f1 < v1)
        for v2 in v2s:
            c2 = (f2 > v2) if d2 == "above" else (f2 < v2)
            for v3 in v3s:
                c3  = (f3 > v3) if d3 == "above" else (f3 < v3)
                acc = accuracy_score(y, (c1 & c2 & c3).astype(int))
                if acc > best_acc:
                    best_acc, best = acc, (v1, v2, v3)
    return best[0], best[1], best[2], best_acc


# ─── LOOCV ────────────────────────────────────────────────────────────────────

ALL_SUBJECTS = [1, 2, 3, 4]

data = load_npy_dataset()
print("Extracting features...")
feats = extract_window_features(data['X'])

print("Building fall clips with angle-rate feature...")
fall_clips_all = build_fall_clips(
    data, feats, ALL_SUBJECTS,
    positive_acts={1, 2, 3, 4, 5},
    mask_acts=set(range(1, 12))
)
print(f"  Total fall clips: {len(fall_clips_all)}")

# Check angle rate distributions
print("\nAngle rate (deg/sec) by activity:")
for act in sorted(fall_clips_all['activity'].unique()):
    m = fall_clips_all['activity'] == act
    v = fall_clips_all[m]['clip_max_angle_rate']
    lbl = 'FALL' if act in {1,2,3,4,5} else 'nofall'
    print(f"  Act{act:02d} {lbl}: p25={np.percentile(v,25):.1f}  "
          f"p50={np.percentile(v,50):.1f}  p75={np.percentile(v,75):.1f}")

run_clips_all = build_clips_simple(
    data, feats, ALL_SUBJECTS,
    positive_acts={8}, mask_acts={6, 8, 9, 11}
)
inact_clips_all = build_clips_simple(
    data, feats, ALL_SUBJECTS,
    positive_acts={7}, mask_acts={6, 7}
)


# ─── LOOCV runners ───────────────────────────────────────────────────────────

def run_loocv(clips_all, task_name, predict_fn):
    accs, precs, recs, f1s = [], [], [], []
    print(f"\n{'='*55}")
    print(f"  {task_name.upper()}")
    print(f"{'='*55}")
    for test_subj in ALL_SUBJECTS:
        tr = clips_all[clips_all['subject'] != test_subj].reset_index(drop=True)
        te = clips_all[clips_all['subject'] == test_subj].reset_index(drop=True)
        if len(te) == 0 or te['label'].sum() == 0 or te['label'].sum() == len(te):
            continue
        pred = predict_fn(tr, te)
        acc  = accuracy_score(te['label'], pred)
        prec = precision_score(te['label'], pred, zero_division=0)
        rec  = recall_score(te['label'], pred, zero_division=0)
        f1   = f1_score(te['label'], pred, zero_division=0)
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"  Test=S{test_subj}: acc={acc*100:.1f}%  "
              f"prec={prec*100:.1f}%  rec={rec*100:.1f}%  f1={f1*100:.1f}%")
    return np.array(accs), np.array(precs), np.array(recs), np.array(f1s)


# ─── FALL predict function ───────────────────────────────────────────────────

def fall_predict(tr, te):
    y_tr = tr['label'].values

    # Feature 1: angle alone
    a1, acc1 = gs1(tr['max_body_angle'].values, y_tr,
                   np.arange(40.0, 110.0, 3.0), direction="above")

    # Feature 2: angle + angle_rate (rate separates falls from lying-down)
    a2, ar2, acc2 = 0.0, 0.0, 0.0
    for a in np.arange(40.0, 110.0, 5.0):
        c1 = tr['max_body_angle'] > a
        for r in np.arange(10.0, 120.0, 5.0):
            c2 = tr['clip_max_angle_rate'] > r
            acc = accuracy_score(y_tr, (c1 & c2).astype(int))
            if acc > acc2:
                acc2, a2, ar2 = acc, a, r

    # Feature 3: angle + vel (original approach)
    a3, v3 = 0.0, 0.0
    acc3 = 0.0
    for a in np.arange(40.0, 110.0, 5.0):
        c1 = tr['max_body_angle'] > a
        for v in np.arange(0.02, 0.80, 0.03):
            c2 = tr['clip_max_vel'] > v
            acc = accuracy_score(y_tr, (c1 & c2).astype(int))
            if acc > acc3:
                acc3, a3, v3 = acc, a, v

    best = max([(acc1, 'angle'), (acc2, 'angle+rate'), (acc3, 'angle+vel')],
               key=lambda x: x[0])

    if best[1] == 'angle':
        return (te['max_body_angle'] > a1).astype(int)
    elif best[1] == 'angle+rate':
        return ((te['max_body_angle'] > a2) & (te['clip_max_angle_rate'] > ar2)).astype(int)
    else:
        return ((te['max_body_angle'] > a3) & (te['clip_max_vel'] > v3)).astype(int)


# ─── RUNNING predict ─────────────────────────────────────────────────────────

def run_predict(tr, te):
    mvt, _ = gs1(tr['mean_kp_disp'].values, tr['label'].values,
                 np.arange(0.0005, 0.015, 0.0005), direction="below")
    return (te['mean_kp_disp'] < mvt).astype(int)


# ─── INACTIVITY predict — midpoint threshold ─────────────────────────────────

def inact_predict(tr, te):
    thr = midpoint_threshold(tr['mean_kp_disp'].values, tr['label'].values,
                             direction="below")
    return (te['mean_kp_disp'] < thr).astype(int)


# ─── Run all ─────────────────────────────────────────────────────────────────

fall_accs,  fall_precs,  fall_recs,  fall_f1s  = run_loocv(fall_clips_all,  "FALL",       fall_predict)
run_accs,   run_precs,   run_recs,   run_f1s   = run_loocv(run_clips_all,   "RUNNING",    run_predict)
inact_accs, inact_precs, inact_recs, inact_f1s = run_loocv(inact_clips_all, "INACTIVITY", inact_predict)


# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  FINAL SUMMARY — LOOCV (4 subjects, UP-Fall dataset)")
print("=" * 60)
print(f"  {'Task':<14} {'Accuracy (mean+/-std)':>22} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'Status'}")
print(f"  {'-'*70}")

all_tasks = [
    ("Fall",       fall_accs,  fall_precs,  fall_recs,  fall_f1s),
    ("Running",    run_accs,   run_precs,   run_recs,   run_f1s),
    ("Inactivity", inact_accs, inact_precs, inact_recs, inact_f1s),
]
for name, accs, precs, recs, f1s in all_tasks:
    if len(accs) == 0: continue
    ok = "[OK] >= 90%" if accs.mean() >= 0.90 else "[!!] Below 90%"
    print(f"  {name:<14} {accs.mean()*100:>6.2f} +/- {accs.std()*100:.1f}%"
          f"  {precs.mean()*100:>5.1f}%"
          f"  {recs.mean()*100:>5.1f}%"
          f"  {f1s.mean()*100:>5.1f}%  {ok}")

all_means = [accs.mean() for _, accs, *_ in all_tasks if len(accs) > 0]
print(f"\n  Overall mean accuracy : {np.mean(all_means)*100:.2f}%")
print(f"  Min task accuracy     : {min(all_means)*100:.2f}%")
print(f"\n  {'All tasks >= 90%!' if all(a >= 0.90 for a in all_means) else 'Some tasks below 90% — see details above'}")
