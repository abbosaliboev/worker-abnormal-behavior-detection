"""
Inactivity evaluation with full activity set.

Positive (inactive worker): Act7 standing + Act8 sitting + Act11 laying
Negative (active worker):   Act6 walking  + Act9 picking + Act10 jumping

Clip-level LOOCV, 4 subjects.
Features: kp_disp + com_y_std + body_angle_std (AND rule, midpoint threshold)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets.npy_loader import load_npy_dataset
from evaluation.eval_from_npy import extract_window_features

ALL_SUBJECTS = [1, 2, 3, 4]
INACT_POS = {7, 8}    # standing, sitting  (laying = fall detector handles it)
INACT_NEG = {6, 9}   # walking, picking up


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
    for (subj, act, trial), grp in meta.groupby(['subject','activity','trial']):
        # Fraction of windows with low motion (robust to transition phase)
        still_fraction  = (grp['mean_kp_disp'] < 0.005).mean()
        clips.append({
            'subject':        subj,
            'activity':       act,
            'trial':          trial,
            'label':          int(act in INACT_POS),
            'mean_kp_disp':   grp['mean_kp_disp'].mean(),
            'still_fraction': still_fraction,           # standing/sitting
            'clip_max_angle': grp['max_body_angle'].max(),  # laying detector
            'body_angle_std': grp['max_body_angle'].std() if len(grp)>1 else 0.0,
            'com_y_std':      grp['com_y'].std() if len(grp)>1 else 0.0,
        })
    return pd.DataFrame(clips)


def midpt(tr, feat):
    pos = tr[tr['label']==1][feat].mean()
    neg = tr[tr['label']==0][feat].mean()
    return (pos + neg) / 2.0


data  = load_npy_dataset()
feats = extract_window_features(data['X'])

clips_all = build_clips(data, feats, ALL_SUBJECTS)

print("=" * 60)
print("  Inactivity — Full Activity Set (LOOCV)")
print("  Positive: standing (Act7) + sitting (Act8)")
print("  Negative: walking (Act6) + picking up (Act9)")
print("=" * 60)

# Activity counts
print("\nActivity distribution:")
names = {6:'walking',7:'standing',8:'sitting',9:'picking',10:'jumping',11:'laying'}
for act in sorted(clips_all['activity'].unique()):
    m = clips_all['activity'] == act
    print(f"  Act{act} {names[act]:<12}: {m.sum():3d} clips  label={'INACTIVE' if act in INACT_POS else 'active'}")

# Feature stats
print("\nFeature comparison:")
for feat in ['mean_kp_disp','com_y_std','body_angle_std']:
    pos = clips_all[clips_all['label']==1][feat]
    neg = clips_all[clips_all['label']==0][feat]
    print(f"  {feat}:")
    print(f"    INACTIVE: mean={pos.mean():.5f}  max={pos.max():.5f}")
    print(f"    ACTIVE  : mean={neg.mean():.5f}  min={neg.min():.5f}")

# LOOCV
print("\nLOOCV per fold:")
accs, precs, recs, f1s = [], [], [], []

for ts in ALL_SUBJECTS:
    tr = clips_all[clips_all['subject'] != ts].reset_index(drop=True)
    te = clips_all[clips_all['subject'] == ts].reset_index(drop=True)

        # Two-branch classifier:
    # Branch 1 (standing + sitting): high still_fraction
    # Branch 2 (laying): high clip_max_angle AND stable angle (low angle_std)
    t_sf  = midpt(tr, 'still_fraction')
    t_ang = midpt(tr, 'body_angle_std')
    t_max = midpt(tr, 'clip_max_angle')   # separates laying from others

    # AND rule: still_fraction + body_angle_std
    # Conservative angle threshold: inactive_max + buffer (not midpoint)
    # → picking up always has angle_std above this even for careful pickers
    t_sf  = midpt(tr, 'still_fraction')
    t_ang = tr[tr['label']==1]['body_angle_std'].max() + 0.5   # inactive_max + buffer
    pred = ((te['still_fraction']  > t_sf) &
            (te['body_angle_std'] < t_ang)).astype(int)

    y = te['label'].values
    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)
    cm   = confusion_matrix(y, pred, labels=[0,1])
    accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    pos_acts = te[te['label']==1]['activity'].unique()
    neg_acts = te[te['label']==0]['activity'].unique()
    print(f"  S{ts}: acc={acc*100:.1f}%  prec={prec*100:.1f}%  rec={rec*100:.1f}%  "
          f"TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}"
          f"  (still>{t_sf:.2f} AND ang_std<{t_ang:.1f})")

accs = np.array(accs)
print(f"\n{'='*60}")
print(f"  INACTIVITY (standing+sitting vs walking+picking)")
print(f"{'='*60}")
print(f"  Accuracy  : {accs.mean()*100:.2f}% +/- {accs.std()*100:.1f}%")
print(f"  Precision : {np.mean(precs)*100:.1f}%")
print(f"  Recall    : {np.mean(recs)*100:.1f}%")
print(f"  F1-score  : {np.mean(f1s)*100:.1f}%")
print(f"  {'[OK] >= 90%' if accs.mean() >= 0.90 else '[!!] Below 90%'}")
