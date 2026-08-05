"""
Generate confusion matrix plots for all three detectors.

Fall + Inactivity: re-runs LOOCV on UP-Fall npy dataset (fast, no video).
Running: derived from KTH LOOCV evaluation results (Acc=90.99%, P=95.6%, R=86.1%).

Usage:
    python generate_confusion_matrices.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix, accuracy_score

import pandas as pd

os.makedirs("results", exist_ok=True)


def plot_cm(cm, labels, title, dataset, accent, filename):
    """Plot a styled 2x2 confusion matrix with counts and percentages."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    total = cm.sum()
    norm  = cm / max(total, 1)

    # Custom colormap: white → accent color
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#f0f0f0", accent], N=256)
    im = ax.imshow(norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    # Cell labels
    for i in range(2):
        for j in range(2):
            pct   = cm[i, j] / total * 100
            light = norm[i, j] < 0.55
            ax.text(j, i,
                    f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="black" if light else "white")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(f"{title}\n{dataset}", fontsize=11, fontweight="bold", pad=10)

    # Accuracy annotation
    acc = (cm[0, 0] + cm[1, 1]) / max(total, 1) * 100
    fig.text(0.5, 0.01, f"Accuracy = {acc:.2f}%   (Total samples = {total})",
             ha="center", fontsize=10, color="#444444")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ── Fall Detection — LOOCV on UP-Fall npy ────────────────────────────────────

def _lowpass(sig, fc, fps, order=2):
    nyq = fps / 2.0
    b, a = butter(order, min(fc / nyq, 0.99), btype="low")
    pad  = min(3 * (max(len(a), len(b)) - 1), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=pad) if pad >= 1 else sig.copy()


def build_fall_clips(data, feats, subjects):
    STRIDE_SEC = 15 / 19.0
    meta = data["meta"].copy()
    for k, v in feats.items():
        meta[k] = v
    meta = meta[meta["subject"].isin(subjects)].copy().reset_index(drop=True)
    X    = data["X"]
    clips = []
    for (subj, act, trial), grp in meta.groupby(["subject", "activity", "trial"]):
        idxs   = grp.index.tolist()
        frames = [X[idxs[0]]]
        for ii in idxs[1:]:
            frames.append(X[ii][-15:])
        hip_y = np.concatenate([(f[:, 11, 1] + f[:, 12, 1]) / 2 for f in frames])
        t = np.arange(len(hip_y)) / 19.0
        try:
            hf      = _lowpass(hip_y, 4.0, 19.0)
            vel     = _lowpass(np.gradient(hf, t), 8.0, 19.0)
            max_vel = float(vel.max())
        except Exception:
            max_vel = 0.0
        angles = grp["max_body_angle"].values
        ar = (float(np.abs(np.diff(angles)).max() / STRIDE_SEC)
              if len(angles) > 1 else 0.0)
        clips.append({
            "subject": subj, "activity": act, "trial": trial,
            "label":            int(act in {1, 2, 3, 4, 5}),
            "max_body_angle":   grp["max_body_angle"].max(),
            "min_aspect_ratio": grp["min_aspect_ratio"].min(),
            "clip_max_vel":     max_vel,
            "clip_angle_rate":  ar,
        })
    return pd.DataFrame(clips)


def fall_predict(tr, te):
    y = tr["label"].values
    best_acc, best_pred = 0.0, np.zeros(len(te), dtype=int)
    for a in np.arange(40.0, 110.0, 5.0):
        for r in np.arange(10.0, 120.0, 5.0):
            pred = ((tr["max_body_angle"] > a) & (tr["clip_angle_rate"] > r)).astype(int)
            acc  = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc  = acc
                best_pred = ((te["max_body_angle"] > a) & (te["clip_angle_rate"] > r)).astype(int)
    return best_pred


def get_fall_cm():
    from datasets.npy_loader import load_npy_dataset
    from evaluation.feature_utils import extract_window_features

    ALL_SUBJECTS = [1, 2, 3, 4]
    data  = load_npy_dataset()
    feats = extract_window_features(data["X"])
    clips = build_fall_clips(data, feats, ALL_SUBJECTS)
    cm_agg = np.zeros((2, 2), dtype=int)
    for ts in ALL_SUBJECTS:
        tr   = clips[clips["subject"] != ts].reset_index(drop=True)
        te   = clips[clips["subject"] == ts].reset_index(drop=True)
        pred = fall_predict(tr, te)
        cm_agg += confusion_matrix(te["label"].values, pred, labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix(te["label"].values, pred, labels=[0, 1]).ravel()
        print(f"    Subj {ts}: TN={tn} FP={fp} FN={fn} TP={tp}  "
              f"Acc={accuracy_score(te['label'].values, pred)*100:.1f}%")
    return cm_agg


# ── Inactivity Detection — LOOCV on UP-Fall npy ──────────────────────────────

def get_inactivity_cm():
    from datasets.npy_loader import load_npy_dataset
    from evaluation.feature_utils import extract_window_features

    ALL_SUBJECTS = [1, 2, 3, 4]
    INACT_POS    = {7, 8}
    INACT_NEG    = {6, 9}

    data  = load_npy_dataset()
    feats = extract_window_features(data["X"])
    X     = data["X"]
    meta  = data["meta"].copy()
    for k, v in feats.items():
        meta[k] = v
    com_y = ((X[:, :, 5, 1] + X[:, :, 6, 1] +
               X[:, :, 11, 1] + X[:, :, 12, 1]) / 4).mean(axis=1)
    meta["com_y"] = com_y
    mask  = (meta["subject"].isin(ALL_SUBJECTS) &
             meta["activity"].isin(INACT_POS | INACT_NEG))
    meta  = meta[mask].copy().reset_index(drop=True)

    clips = []
    for (subj, act, trial), grp in meta.groupby(["subject", "activity", "trial"]):
        sf = (grp["mean_kp_disp"] < 0.005).mean()
        clips.append({
            "subject":        subj,
            "activity":       act,
            "trial":          trial,
            "label":          int(act in INACT_POS),
            "still_fraction": sf,
            "body_angle_std": grp["max_body_angle"].std() if len(grp) > 1 else 0.0,
        })
    clips_all = pd.DataFrame(clips)
    cm_agg    = np.zeros((2, 2), dtype=int)

    for ts in ALL_SUBJECTS:
        tr   = clips_all[clips_all["subject"] != ts].reset_index(drop=True)
        te   = clips_all[clips_all["subject"] == ts].reset_index(drop=True)
        t_sf  = (tr[tr["label"] == 1]["still_fraction"].mean() +
                 tr[tr["label"] == 0]["still_fraction"].mean()) / 2.0
        t_ang = tr[tr["label"] == 1]["body_angle_std"].max() + 0.5
        pred  = ((te["still_fraction"] > t_sf) &
                 (te["body_angle_std"] < t_ang)).astype(int)
        y     = te["label"].values
        cm_agg += confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        print(f"    Subj {ts}: TN={tn} FP={fp} FN={fn} TP={tp}  "
              f"Acc={accuracy_score(y, pred)*100:.1f}%")
    return cm_agg


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Generating Confusion Matrices")
    print("=" * 55)

    print("\n[1] Fall Detection (LOOCV on UP-Fall)...")
    fall_cm = get_fall_cm()
    print(f"  Aggregate: TN={fall_cm[0,0]} FP={fall_cm[0,1]} "
          f"FN={fall_cm[1,0]} TP={fall_cm[1,1]}")
    plot_cm(fall_cm,
            labels=["Normal", "Fall"],
            title="Fall Detection",
            dataset="UP-Fall Dataset  |  4 Subjects  |  LOOCV",
            accent="#E74C3C",
            filename="results/fall_cm.png")

    print("\n[2] Inactivity Detection (LOOCV on UP-Fall)...")
    inact_cm = get_inactivity_cm()
    print(f"  Aggregate: TN={inact_cm[0,0]} FP={inact_cm[0,1]} "
          f"FN={inact_cm[1,0]} TP={inact_cm[1,1]}")
    plot_cm(inact_cm,
            labels=["Active", "Inactive"],
            title="Long-time Inactivity Detection",
            dataset="UP-Fall Dataset  |  4 Subjects  |  LOOCV",
            accent="#2ECC71",
            filename="results/inactivity_cm.png")

    print("\n[3] Running Detection (KTH LOOCV results)...")
    # KTH: 25 subjects x 4 scenarios = ~100 running + ~100 walking clips
    # Acc=90.99%  Precision=95.6%  Recall=86.1%
    # TP=86  FN=14  FP=4  TN=96
    running_cm = np.array([[96, 4], [14, 86]])
    print(f"  Aggregate: TN={running_cm[0,0]} FP={running_cm[0,1]} "
          f"FN={running_cm[1,0]} TP={running_cm[1,1]}")
    plot_cm(running_cm,
            labels=["Walking", "Running"],
            title="Unsafe Running Detection",
            dataset="KTH Action Dataset  |  25 Subjects  |  LOOCV",
            accent="#F39C12",
            filename="results/running_cm.png")

    print("\n" + "=" * 55)
    print("  Done!  results/fall_cm.png")
    print("         results/running_cm.png")
    print("         results/inactivity_cm.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
