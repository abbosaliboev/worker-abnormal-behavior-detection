"""
Leave-One-Out Cross-Validation across 4 subjects.

For each fold: train on 3 subjects, test on 1 subject.
Reports mean ± std accuracy across 4 folds.

This is more statistically reliable than a fixed 2/2 split
when only 4 subjects are available.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets.npy_loader import load_npy_dataset
from evaluation.eval_from_npy import extract_window_features


def build_clips_for_subject(data, feats, subject_ids, positive_acts, mask_acts=None):
    import pandas as pd
    meta = data['meta'].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta['subject'].isin(subject_ids)]
    if mask_acts:
        meta = meta[meta['activity'].isin(mask_acts)].reset_index(drop=True)
    clips = []
    for (subj, act, trial), grp in meta.groupby(['subject', 'activity', 'trial']):
        clips.append({
            'subject': subj, 'activity': act, 'trial': trial,
            'label': int(act in positive_acts),
            'max_body_angle':   grp['max_body_angle'].max(),
            'min_aspect_ratio': grp['min_aspect_ratio'].min(),
            'max_hip_y':        grp['max_hip_y'].max(),
            'mean_kp_disp':     grp['mean_kp_disp'].mean(),
            'mean_vert_osc':    grp['vert_osc'].mean(),
        })
    import pandas as pd
    return pd.DataFrame(clips)


def gs1(feat, y, vals, direction="above"):
    best_acc, best_v = 0.0, vals[0]
    for v in vals:
        p = (feat > v) if direction == "above" else (feat < v)
        acc = accuracy_score(y, p.astype(int))
        if acc > best_acc:
            best_acc, best_v = acc, v
    return best_v, best_acc


def gs_or(f1, f2, y, v1s, v2s):
    best_acc, best = 0.0, (v1s[0], v2s[0])
    for v1 in v1s:
        c1 = f1 > v1
        for v2 in v2s:
            c2 = f2 < v2
            acc = accuracy_score(y, (c1 | c2).astype(int))
            if acc > best_acc:
                best_acc, best = acc, (v1, v2)
    return best[0], best[1], best_acc


def evaluate_fold(train_clips, test_clips, task):
    y_tr = train_clips['label'].values
    y_te = test_clips['label'].values

    if task == "fall":
        # Try angle, ar, and OR combination
        a1, _ = gs1(train_clips['max_body_angle'].values, y_tr,
                    np.arange(40.0, 110.0, 3.0), direction="above")
        a2, ar, _ = gs_or(train_clips['max_body_angle'], train_clips['min_aspect_ratio'],
                          y_tr, np.arange(40.0, 110.0, 5.0), np.arange(0.10, 0.90, 0.05))

        # Evaluate both on test, pick best by train accuracy (already done via gs)
        p1 = (test_clips['max_body_angle'] > a1).astype(int)
        p2 = ((test_clips['max_body_angle'] > a2) | (test_clips['min_aspect_ratio'] < ar)).astype(int)
        # Use OR version (more coverage)
        pred = p2

    elif task == "running":
        mvt, _ = gs1(train_clips['mean_kp_disp'].values, y_tr,
                     np.arange(0.0005, 0.015, 0.0005), direction="below")
        pred = (test_clips['mean_kp_disp'] < mvt).astype(int)

    elif task == "inactivity":
        mvt, _ = gs1(train_clips['mean_kp_disp'].values, y_tr,
                     np.arange(0.0005, 0.030, 0.0005), direction="below")
        pred = (test_clips['mean_kp_disp'] < mvt).astype(int)

    acc  = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    f1   = f1_score(y_te, pred, zero_division=0)
    return acc, prec, rec, f1


ALL_SUBJECTS = [1, 2, 3, 4]
TASKS = {
    "fall":       {"pos": {1,2,3,4,5,10}, "mask": set(range(1,12))},
    "running":    {"pos": {8},             "mask": {6,8,9,11}},
    "inactivity": {"pos": {7,10},          "mask": {6,7,9,10,11}},
}

print("=" * 60)
print("  Leave-One-Out CV  (4 subjects, UP-Fall dataset)")
print("=" * 60)

# Load all data
data = load_npy_dataset()
print("Extracting features (all subjects)...")
feats = extract_window_features(data['X'])

all_results = {task: {"acc": [], "prec": [], "rec": [], "f1": []} for task in TASKS}

for test_subj in ALL_SUBJECTS:
    train_subjs = [s for s in ALL_SUBJECTS if s != test_subj]
    print(f"\n--- Fold: test=Subject{test_subj}  train=Subjects{train_subjs} ---")

    for task, cfg in TASKS.items():
        tr_clips = build_clips_for_subject(data, feats, train_subjs, cfg["pos"], cfg["mask"])
        te_clips = build_clips_for_subject(data, feats, [test_subj],  cfg["pos"], cfg["mask"])

        if len(te_clips) == 0 or te_clips['label'].sum() == 0 or te_clips['label'].sum() == len(te_clips):
            print(f"  {task}: skipped (degenerate split)")
            continue

        acc, prec, rec, f1 = evaluate_fold(tr_clips, te_clips, task)
        all_results[task]["acc"].append(acc)
        all_results[task]["prec"].append(prec)
        all_results[task]["rec"].append(rec)
        all_results[task]["f1"].append(f1)
        print(f"  {task:<12}: acc={acc*100:.1f}%  prec={prec*100:.1f}%  rec={rec*100:.1f}%  f1={f1*100:.1f}%")

print("\n" + "=" * 60)
print("  LOOCV SUMMARY (mean ± std across folds)")
print("=" * 60)
print(f"  {'Task':<14} {'Accuracy':>12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*54}")
for task, r in all_results.items():
    if not r["acc"]:
        continue
    accs = np.array(r["acc"])
    prec = np.array(r["prec"])
    recs = np.array(r["rec"])
    f1s  = np.array(r["f1"])
    ok = "[OK]" if accs.mean() >= 0.90 else "[!!]"
    print(f"  {task:<14} {accs.mean()*100:>6.2f}±{accs.std()*100:.1f}% "
          f"{prec.mean()*100:>9.1f}% "
          f"{recs.mean()*100:>7.1f}% "
          f"{f1s.mean()*100:>7.1f}%  {ok}")

mean_acc = np.mean([np.mean(r["acc"]) for r in all_results.values() if r["acc"]])
print(f"\n  Mean accuracy (all tasks): {mean_acc*100:.2f}%")
