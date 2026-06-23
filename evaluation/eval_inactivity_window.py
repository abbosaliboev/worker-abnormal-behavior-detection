"""
Window-level inactivity evaluation (harder, more credible than clip-level).

Negative class: Act6 + Act8 + Act9 + Act11
  Act8 (running) is INCLUDED as negative - in this dataset it looks like
  standing (low motion), making classification harder and more realistic.
  A real system uses the cascade, but this evaluation shows raw detector performance.

Each 30-frame window evaluated independently (not averaged per clip).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from datasets.npy_loader import load_npy_dataset
from evaluation.eval_from_npy import extract_window_features

ALL_SUBJECTS = [1, 2, 3, 4]
INACT_POS  = {7}               # standing still
INACT_NEG  = {6, 8, 9, 11}    # walking + running + jumping + picking
# running (Act8) included as negative — makes problem harder and more realistic


def midpoint(pos_vals, neg_vals):
    return (pos_vals.mean() + neg_vals.mean()) / 2.0


data  = load_npy_dataset()
feats = extract_window_features(data['X'])
acts  = data['meta']['activity'].values
subjs = data['meta']['subject'].values

# Window-level labels
mask = np.isin(acts, list(INACT_POS | INACT_NEG))
y_all    = np.isin(acts[mask], list(INACT_POS)).astype(int)
subj_all = subjs[mask]

# Features per window
disp_all  = feats['mean_kp_disp'][mask]
osc_all   = feats['vert_osc'][mask]       # CoM vertical oscillation
angle_all = feats['max_body_angle'][mask]  # used for body_angle_std proxy

print("=" * 60)
print("  Inactivity — Window-Level LOOCV")
print(f"  Positive: Act7 (standing)   | windows={y_all.sum()}")
print(f"  Negative: Act6+8+9+11       | windows={(y_all==0).sum()}")
print("=" * 60)

# Feature stats
print("\nPer-window feature stats:")
for name, feat in [("kp_disp", disp_all), ("vert_osc", osc_all),
                   ("body_angle", angle_all)]:
    pos = feat[y_all == 1]
    neg = feat[y_all == 0]
    print(f"  {name}:")
    print(f"    POS (stand): mean={pos.mean():.5f}  p50={np.median(pos):.5f}")
    print(f"    NEG (other): mean={neg.mean():.5f}  p50={np.median(neg):.5f}")

# LOOCV
accs, precs, recs, f1s = [], [], [], []

print("\nPer-fold results:")
for test_subj in ALL_SUBJECTS:
    tr_mask = subj_all != test_subj
    te_mask = subj_all == test_subj

    y_tr   = y_all[tr_mask];    y_te   = y_all[te_mask]
    d_tr   = disp_all[tr_mask]; d_te   = disp_all[te_mask]
    o_tr   = osc_all[tr_mask];  o_te   = osc_all[te_mask]
    an_tr  = angle_all[tr_mask];an_te  = angle_all[te_mask]

    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        continue

    # Midpoint thresholds (robust to subject variability)
    t_disp  = midpoint(d_tr[y_tr==1],  d_tr[y_tr==0])
    t_osc   = midpoint(o_tr[y_tr==1],  o_tr[y_tr==0])
    t_angle = midpoint(an_tr[y_tr==1], an_tr[y_tr==0])

    # AND rule: standing = low displacement AND low oscillation AND low angle
    pred = ((d_te  < t_disp) &
            (o_te  < t_osc)  &
            (an_te < t_angle)).astype(int)

    acc  = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    f1   = f1_score(y_te, pred, zero_division=0)
    cm   = confusion_matrix(y_te, pred, labels=[0, 1])
    accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    n_pos = y_te.sum(); n_neg = (y_te==0).sum()
    print(f"  S{test_subj} (pos={n_pos} neg={n_neg} windows): "
          f"acc={acc*100:.1f}%  prec={prec*100:.1f}%  "
          f"rec={rec*100:.1f}%  f1={f1*100:.1f}%"
          f"  (TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]})")

accs = np.array(accs)
print(f"\n{'='*60}")
print(f"  INACTIVITY (window-level, Act6+8+9+11 negative)")
print(f"{'='*60}")
print(f"  Accuracy  : {accs.mean()*100:.2f}% +/- {accs.std()*100:.1f}%")
print(f"  Precision : {np.mean(precs)*100:.1f}%")
print(f"  Recall    : {np.mean(recs)*100:.1f}%")
print(f"  F1-score  : {np.mean(f1s)*100:.1f}%")
print(f"  {'[OK] >= 90%' if accs.mean() >= 0.90 else '[!!] Below 90%'}")

print(f"\nNote: Act8 (running) included in negative class.")
print(f"In production, the cascade (running detector -> inactivity) avoids this.")
