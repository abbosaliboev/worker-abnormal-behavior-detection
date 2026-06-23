"""
Load pre-extracted X.npy / y.npy / meta.csv for evaluation.

X.npy  : (N, T, V, C) = (4479, 30, 17, 3)  — YOLO keypoints per window
y.npy  : (N,)                               — 1=fall, 0=no-fall
meta.csv: seq_id, subject, activity, trial, start_frame, label

This loader adds per-task label vectors so we can evaluate all three
detectors without re-processing any video frames.

Activity → task mapping:
  Fall       : activities 1-5  (all labeled 1 in y.npy)
  Running    : activity 8      (non-fall in y.npy, but is_running=True)
  Inactivity : activities 7,10 (standing / lying motionless)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    NPY_DATA_DIR,
    FALL_ACTIVITY_IDS, RUNNING_ACTIVITY_IDS, INACTIVITY_ACTIVITY_IDS,
    EVAL,
)


def load_npy_dataset(data_dir: str = NPY_DATA_DIR) -> dict:
    """
    Returns a dict with:
      X         : (N, T, V, C) float32
      y_fall    : (N,) int — 1 if activity is a fall type
      y_running : (N,) int — 1 if activity is running
      y_inactivity: (N,) int — 1 if activity is standing/lying still
      meta      : DataFrame with columns [seq_id, subject, activity, trial, ...]
    """
    X    = np.load(os.path.join(data_dir, "X.npy"))      # (N, T, V, C)
    y    = np.load(os.path.join(data_dir, "y.npy"))      # (N,) fall label
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv"))

    acts = meta["activity"].values

    y_fall       = (y == 1).astype(int)
    y_running    = np.isin(acts, list(RUNNING_ACTIVITY_IDS)).astype(int)
    y_inactivity = np.isin(acts, list(INACTIVITY_ACTIVITY_IDS)).astype(int)

    print(f"Dataset loaded: N={len(X)}  T={X.shape[1]}  V={X.shape[2]}  C={X.shape[3]}")
    print(f"  Fall:       pos={y_fall.sum():4d}  neg={(y_fall==0).sum():4d}")
    print(f"  Running:    pos={y_running.sum():4d}  neg={(y_running==0).sum():4d}")
    print(f"  Inactivity: pos={y_inactivity.sum():4d}  neg={(y_inactivity==0).sum():4d}")

    return {
        "X":            X,
        "y_fall":       y_fall,
        "y_running":    y_running,
        "y_inactivity": y_inactivity,
        "meta":         meta,
    }


def train_test_split(data: dict) -> tuple[dict, dict]:
    """
    Split by subject ID so no subject appears in both train and test.
    Returns (train_data, test_data) with the same keys.
    """
    subjects = data["meta"]["subject"].values
    N        = len(data["X"])
    idx      = np.arange(N)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=EVAL["test_subjects_fraction"],
        random_state=EVAL["random_seed"],
    )
    train_idx, test_idx = next(gss.split(idx, groups=subjects))

    def _subset(d, ids):
        return {
            "X":            d["X"][ids],
            "y_fall":       d["y_fall"][ids],
            "y_running":    d["y_running"][ids],
            "y_inactivity": d["y_inactivity"][ids],
            "meta":         d["meta"].iloc[ids].reset_index(drop=True),
        }

    train = _subset(data, train_idx)
    test  = _subset(data, test_idx)

    train_subj = set(train["meta"]["subject"].unique())
    test_subj  = set(test["meta"]["subject"].unique())
    print(f"Train: {len(train_idx)} windows, subjects {sorted(train_subj)}")
    print(f"Test:  {len(test_idx)}  windows, subjects {sorted(test_subj)}")
    return train, test


def group_into_sequences(data: dict, task: str) -> list[dict]:
    """
    Group windows by (subject, activity, trial) into contiguous sequences.
    Used for inactivity detection which needs longer time windows.

    Returns list of dicts: {X_seq: (T_total, V, C), label: 0|1, meta: {...}}
    """
    meta = data["meta"]
    label_col = {"fall": "y_fall", "running": "y_running",
                 "inactivity": "y_inactivity"}[task]
    labels = data[label_col]
    X      = data["X"]    # (N, T, V, C)

    sequences = []
    for (subj, act, trial), grp in meta.groupby(["subject", "activity", "trial"]):
        ids   = grp.index.tolist()
        # Concatenate windows along time (non-overlapping: take first window fully,
        # then only the new frames from each subsequent window given stride=15)
        frames_list = [X[ids[0]]]                    # full first window (30 frames)
        for i in ids[1:]:
            frames_list.append(X[i][-15:])            # only last 15 (new) frames
        X_seq = np.concatenate(frames_list, axis=0)  # (T_total, V, C)
        lbl   = int(labels[ids[0]])
        sequences.append({
            "X_seq":    X_seq,
            "label":    lbl,
            "subject":  subj,
            "activity": act,
            "trial":    trial,
        })

    return sequences
