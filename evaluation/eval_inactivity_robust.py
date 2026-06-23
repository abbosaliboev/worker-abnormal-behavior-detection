"""
Robust Inactivity Detection: Multi-feature + Spatial Tracking (clip-level).

Negative class: Act6 (walking) + Act9 (jumping) + Act11 (picking up)
  Act8 (running) -> handled by running detector  (cascade)
  Act10 (lying)  -> handled by fall detector      (cascade)

Features (all computed at CLIP level from per-window aggregates):
  1. mean_kp_disp       -- mean keypoint displacement per frame
  2. com_y_std          -- vertical CoM oscillation across windows (CoM of each window)
  3. com_x_range        -- horizontal spatial tracking (CoM X drift across windows)
  4. body_angle_std     -- posture stability (body angle std across windows)

Each feature normalized per training fold; composite score thresholded.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from datasets.npy_loader import load_npy_dataset
from evaluation.eval_from_npy import extract_window_features

ALL_SUBJECTS = [1, 2, 3, 4]
INACT_POS  = {7}
INACT_NEG  = {6, 9, 11}

KP = {"l_shoulder":5,"r_shoulder":6,"l_hip":11,"r_hip":12}


# ─── Per-window CoM (x, y) ───────────────────────────────────────────────────

def compute_window_coms(X_windows: np.ndarray) -> tuple:
    """
    X_windows: (N, T, 17, 3)
    Returns com_x (N,) and com_y (N,) -- mean CoM position per window.
    """
    L, R = KP["l_shoulder"], KP["r_shoulder"]
    H, I = KP["l_hip"], KP["r_hip"]
    xy = X_windows[:, :, :, :2]   # (N, T, 17, 2)
    com_y = (xy[:, :, L, 1] + xy[:, :, R, 1] +
             xy[:, :, H, 1] + xy[:, :, I, 1]).mean(axis=1) / 4.0   # (N,)
    com_x = (xy[:, :, L, 0] + xy[:, :, R, 0] +
             xy[:, :, H, 0] + xy[:, :, I, 0]).mean(axis=1) / 4.0   # (N,)
    return com_x, com_y


# ─── Build clip table ─────────────────────────────────────────────────────────

def build_clips(data, feats, subjects, mask_acts):
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v

    # add per-window CoM
    com_x_w, com_y_w = compute_window_coms(data['X'])
    meta['com_x'] = com_x_w
    meta['com_y'] = com_y_w

    meta = meta[meta['subject'].isin(subjects) &
                meta['activity'].isin(mask_acts)].copy()
    meta.reset_index(drop=True, inplace=True)

    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject','activity','trial']):
        clips.append({
            'subject':        subj,
            'activity':       act,
            'trial':          trial,
            'label':          int(act in INACT_POS),
            # Feature 1: mean per-window keypoint displacement (no boundary artifacts)
            'mean_kp_disp':   grp['mean_kp_disp'].mean(),
            # Feature 2: CoM vertical oscillation across windows
            'com_y_std':      grp['com_y'].std() if len(grp) > 1 else 0.0,
            # Feature 3: CoM horizontal tracking range across windows
            'com_x_range':    grp['com_x'].max() - grp['com_x'].min(),
            # Feature 4: body angle stability across windows
            'body_angle_std': grp['max_body_angle'].std() if len(grp) > 1 else 0.0,
        })
    return pd.DataFrame(clips)


# ─── Threshold search ─────────────────────────────────────────────────────────

def gs1(feat, y, vals, direction="below"):
    best_acc, best_v = 0.0, vals[0]
    for v in vals:
        p = (feat < v) if direction == "below" else (feat > v)
        acc = accuracy_score(y, p.astype(int))
        if acc > best_acc:
            best_acc, best_v = acc, v
    return best_v, best_acc


def midpoint_thr(tr_clips, feat, pos_label=1):
    """
    Threshold = midpoint between positive and negative class means.
    More robust than minimum threshold: generalizes across subjects.
    """
    pos_mean = tr_clips[tr_clips['label'] == pos_label][feat].mean()
    neg_mean = tr_clips[tr_clips['label'] != pos_label][feat].mean()
    return (pos_mean + neg_mean) / 2.0


def and_predict(tr, te):
    """
    3-condition AND rule with midpoint thresholds.
    Each condition tuned on the training split independently:
      1. kp_disp     < midpoint(stand_mean, walk_mean)   -> rejects walking/picking
      2. com_y_std   < midpoint(stand_mean, jump_mean)   -> rejects jumping
      3. body_angle_std < midpoint(stand_mean, pick_mean) -> rejects picking/jumping

    Standing meets ALL three -> INACTIVE.
    """
    # Midpoint thresholds from training data
    thr_disp  = midpoint_thr(tr, 'mean_kp_disp')
    thr_y_std = midpoint_thr(tr, 'com_y_std')
    thr_ang   = midpoint_thr(tr, 'body_angle_std')

    # Apply AND rule to test
    c1 = te['mean_kp_disp']   < thr_disp
    c2 = te['com_y_std']      < thr_y_std
    c3 = te['body_angle_std'] < thr_ang
    pred = (c1 & c2 & c3).astype(int)

    tr_pred = ((tr['mean_kp_disp']   < thr_disp) &
               (tr['com_y_std']      < thr_y_std) &
               (tr['body_angle_std'] < thr_ang)).astype(int)
    tr_acc = accuracy_score(tr['label'].values, tr_pred)

    return pred, (thr_disp, thr_y_std, thr_ang), tr_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

data = load_npy_dataset()
print("Extracting window features...")
feats = extract_window_features(data['X'])

print("Building clip-level tracking features...")
clips_all = build_clips(data, feats, ALL_SUBJECTS, INACT_POS | INACT_NEG)
print(f"  Clips: {len(clips_all)}  pos={clips_all['label'].sum()}  "
      f"neg={(clips_all['label']==0).sum()}")

print("\nClip-level feature distributions:")
names = {6:'walking',7:'standing',9:'jumping',11:'picking'}
for act in sorted(clips_all['activity'].unique()):
    row = clips_all[clips_all['activity'] == act]
    print(f"  Act{act} {names.get(act,'?'):<10}:"
          f"  kp={row['mean_kp_disp'].mean():.5f}"
          f"  com_y={row['com_y_std'].mean():.4f}"
          f"  com_x={row['com_x_range'].mean():.4f}"
          f"  ang={row['body_angle_std'].mean():.2f}")

print("\n" + "="*60)
print("  LOOCV (4-fold, leave one subject out)")
print("="*60)

accs, precs, recs, f1s = [], [], [], []

for test_subj in ALL_SUBJECTS:
    tr = clips_all[clips_all['subject'] != test_subj].reset_index(drop=True)
    te = clips_all[clips_all['subject'] == test_subj].reset_index(drop=True)

    if te['label'].sum() == 0 or te['label'].sum() == len(te):
        print(f"  S{test_subj}: skipped"); continue

    pred, thresholds, tr_acc = and_predict(tr, te)
    y_te = te['label'].values

    acc  = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    f1   = f1_score(y_te, pred, zero_division=0)
    cm   = confusion_matrix(y_te, pred, labels=[0, 1])
    accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    print(f"  S{test_subj}: acc={acc*100:.1f}%  prec={prec*100:.1f}%"
          f"  rec={rec*100:.1f}%  f1={f1*100:.1f}%"
          f"  (thresholds: kp<{thresholds[0]:.4f} y_std<{thresholds[1]:.4f}"
          f" ang<{thresholds[2]:.1f})"
          f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]})")

accs = np.array(accs)
print(f"\n{'='*60}")
print(f"  INACTIVITY RESULTS (multi-feature + tracking)")
print(f"{'='*60}")
print(f"  Accuracy  : {accs.mean()*100:.2f}% +/- {accs.std()*100:.1f}%")
print(f"  Precision : {np.mean(precs)*100:.1f}%")
print(f"  Recall    : {np.mean(recs)*100:.1f}%")
print(f"  F1-score  : {np.mean(f1s)*100:.1f}%")
print(f"  {'[OK] >= 90%' if accs.mean() >= 0.90 else '[!!] Below 90%'}")
print(f"\nNegative class: Act6+9+11 (walking, jumping, picking up)")
print(f"Cascade: Act8 handled by running detector, Act10 by fall detector")
