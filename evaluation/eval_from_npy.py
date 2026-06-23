"""
Fast evaluation of all three rule-based detectors on pre-extracted X.npy data.

Strategy for speed:
  - Pre-compute aggregate features for every window (vectorized, one pass)
  - Grid-search thresholds directly on feature arrays (milliseconds)
  - Full per-frame simulation only during final test evaluation

Usage:
    python -m evaluation.eval_from_npy
    python -m evaluation.eval_from_npy --tune
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from scipy.signal import butter, filtfilt

from src.config import STGCN_FPS, KP, KP_VISIBILITY_THRESHOLD, FALL, RUNNING, INACTIVITY
from src.pose_extractor import pose_from_npy
from src.fall_detector import FallDetector
from src.running_detector import RunningDetector
from src.inactivity_detector import InactivityDetector
from datasets.npy_loader import load_npy_dataset, train_test_split, group_into_sequences


# =============================================================================
# Fast vectorized feature extraction (run ONCE per dataset)
# =============================================================================

def _angle_deg(v1, v2):
    """Angle between v1 and v2 in degrees. Vectorized over last axis."""
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos = np.einsum("...i,...i->...", v1, v2) / (n1[..., 0] * n2[..., 0] + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _joint_angle_seq(seq, a, b, c):
    """Angle at joint b over sequence. seq: (T, 17, 3)."""
    ba = seq[:, a, :2] - seq[:, b, :2]
    bc = seq[:, c, :2] - seq[:, b, :2]
    return _angle_deg(ba, bc)


def extract_window_features(X: np.ndarray, fps: float = STGCN_FPS) -> dict:
    """
    X: (N, T, V, C)  where C = (x, y, conf)

    Returns dict of (N,) arrays with per-window aggregate features.
    Takes ~2-3 seconds for N=4479.
    """
    N, T, V, C = X.shape
    dt = 1.0 / fps

    # Joint shortcuts
    L_SH, R_SH = KP["l_shoulder"], KP["r_shoulder"]
    L_HIP, R_HIP = KP["l_hip"], KP["r_hip"]
    L_KN, R_KN = KP["l_knee"], KP["r_knee"]
    L_AN, R_AN = KP["l_ankle"], KP["r_ankle"]

    xy = X[..., :2]   # (N, T, V, 2)
    cf = X[..., 2]    # (N, T, V)    confidence

    # Midpoints
    mid_hip      = (xy[:, :, L_HIP] + xy[:, :, R_HIP]) / 2.0   # (N, T, 2)
    mid_shoulder = (xy[:, :, L_SH]  + xy[:, :, R_SH])  / 2.0   # (N, T, 2)

    # --- Fall features ---
    # Body tilt angle
    spine    = mid_shoulder - mid_hip                            # (N, T, 2)
    vertical = np.array([0.0, -1.0])
    body_angle = _angle_deg(spine, np.broadcast_to(vertical, spine.shape))  # (N, T)

    # Hip Y (normalized, downward positive)
    hip_y = mid_hip[:, :, 1]   # (N, T)

    # Aspect ratio  (min bbox height / bbox width across visible joints)
    vis_mask = cf >= KP_VISIBILITY_THRESHOLD   # (N, T, V) bool
    # Use max/min of x,y weighted by visibility
    x_vals = np.where(vis_mask, xy[..., 0], np.nan)
    y_vals = np.where(vis_mask, xy[..., 1], np.nan)
    bbox_h = np.nanmax(y_vals, axis=2) - np.nanmin(y_vals, axis=2)  # (N, T)
    bbox_w = np.nanmax(x_vals, axis=2) - np.nanmin(x_vals, axis=2)  # (N, T)
    aspect_ratio = np.where(bbox_w > 1e-4, bbox_h / (bbox_w + 1e-8), 5.0)  # (N, T)

    # Hip downward velocity (finite difference on filtered hip_y)
    hip_vel = np.gradient(hip_y, dt, axis=1)   # (N, T)

    # Per-window aggregates
    max_body_angle  = body_angle.max(axis=1)           # (N,)
    max_hip_y       = hip_y.max(axis=1)                # (N,)
    min_aspect_ratio= aspect_ratio.min(axis=1)         # (N,)
    max_hip_vel     = hip_vel.max(axis=1)              # (N,) downward velocity

    # --- Running features ---
    CoM = (mid_hip + mid_shoulder) / 2.0               # (N, T, 2)
    com_x = CoM[:, :, 0]
    com_y = CoM[:, :, 1]

    # Horizontal speed
    com_dx = np.abs(np.diff(com_x, axis=1))            # (N, T-1)
    horiz_speed = com_dx.mean(axis=1)                  # (N,)

    # Vertical oscillation
    vert_osc = com_y.std(axis=1)                       # (N,)

    # Knee angle difference -> step frequency via zero-crossings
    l_knee = _joint_angle_seq_batch(X, L_HIP, L_KN, L_AN)  # (N, T)
    r_knee = _joint_angle_seq_batch(X, R_HIP, R_KN, R_AN)  # (N, T)
    knee_diff = l_knee - r_knee                             # (N, T)
    sign_changes = np.diff(np.sign(knee_diff), axis=1)     # (N, T-1)
    zc_count = (sign_changes != 0).sum(axis=1)             # (N,)
    step_freq = zc_count / (2.0 * T / fps)                 # (N,) in Hz

    knee_var = knee_diff.var(axis=1)                       # (N,)

    # --- Inactivity features ---
    kp_disp = np.linalg.norm(np.diff(xy, axis=1), axis=-1)  # (N, T-1, V)
    # Only visible joints
    vis_both = vis_mask[:, :-1] & vis_mask[:, 1:]            # (N, T-1, V)
    kp_disp_masked = np.where(vis_both, kp_disp, np.nan)
    mean_disp = np.nanmean(kp_disp_masked, axis=(1, 2))      # (N,)
    mean_disp = np.where(np.isnan(mean_disp), 0.0, mean_disp)

    return {
        # fall
        "max_body_angle":   max_body_angle,
        "max_hip_y":        max_hip_y,
        "min_aspect_ratio": min_aspect_ratio,
        "max_hip_vel":      max_hip_vel,
        # running
        "step_freq":        step_freq,
        "horiz_speed":      horiz_speed,
        "vert_osc":         vert_osc,
        "knee_var":         knee_var,
        # inactivity
        "mean_kp_disp":     mean_disp,
    }


def _joint_angle_seq_batch(X, a, b, c):
    """Angle at joint b for all windows. X: (N, T, V, C). Returns (N, T)."""
    ba = X[:, :, a, :2] - X[:, :, b, :2]
    bc = X[:, :, c, :2] - X[:, :, b, :2]
    return _angle_deg(ba, bc)


# =============================================================================
# Threshold calibration (vectorized, milliseconds)
# =============================================================================

def tune_fall(feats: dict, y: np.ndarray) -> dict:
    """Grid-search on pre-computed features. Returns best thresholds."""
    best_acc = 0.0
    best = {"angle": FALL["angle_threshold"], "hip": FALL["hip_height_threshold"]}

    for angle in np.arange(30.0, 85.0, 2.5):
        for hip in np.arange(0.40, 0.90, 0.025):
            cond_angle = feats["max_body_angle"] > angle
            cond_low   = (feats["max_hip_y"] > hip) | \
                         (feats["min_aspect_ratio"] < FALL["aspect_ratio_threshold"])
            pred = (cond_angle & cond_low).astype(int)
            acc = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc = acc
                best = {"angle": angle, "hip": hip}

    print(f"[Tune Fall]    angle={best['angle']:.1f}deg  hip={best['hip']:.3f}"
          f"  train_acc={best_acc*100:.2f}%")
    return best


def tune_running(feats: dict, y: np.ndarray) -> dict:
    """Grid-search on pre-computed features."""
    best_acc = 0.0
    best = {"freq": RUNNING["step_freq_threshold"],
            "speed": RUNNING["horizontal_speed_threshold"]}

    for freq in np.arange(0.5, 5.0, 0.25):
        for spd in np.arange(0.001, 0.06, 0.002):
            c_freq  = feats["step_freq"]  >= freq
            c_speed = feats["horiz_speed"] >= spd
            c_osc   = feats["vert_osc"]   >= RUNNING["vertical_oscillation_threshold"]
            c_kvar  = feats["knee_var"]   >= RUNNING["knee_angle_variance_threshold"]
            n_cond  = c_freq.astype(int) + c_speed.astype(int) + \
                      c_osc.astype(int)  + c_kvar.astype(int)
            pred = (n_cond >= RUNNING["min_conditions"]).astype(int)
            acc = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc = acc
                best = {"freq": freq, "speed": spd}

    print(f"[Tune Running] freq={best['freq']:.2f}Hz  speed={best['speed']:.4f}"
          f"  train_acc={best_acc*100:.2f}%")
    return best


def tune_inactivity(feats: dict, y: np.ndarray) -> dict:
    """Grid-search movement threshold."""
    best_acc = 0.0
    best = {"mvt": INACTIVITY["movement_threshold"]}

    for mvt in np.arange(0.0005, 0.03, 0.0005):
        pred = (feats["mean_kp_disp"] < mvt).astype(int)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc
            best = {"mvt": mvt}

    print(f"[Tune Inact]   movement={best['mvt']:.5f}"
          f"  train_acc={best_acc*100:.2f}%")
    return best


# =============================================================================
# Full per-frame simulation for final evaluation
# =============================================================================

def evaluate_fall(test_data: dict) -> dict:
    X, y = test_data["X"], test_data["y_fall"]
    y_pred = []
    for i in tqdm(range(len(X)), desc="  Fall eval", unit="win"):
        det = FallDetector(fps=STGCN_FPS)
        fired = 0
        for t in range(len(X[i])):
            pf = pose_from_npy(X[i][t], frame_idx=t, fps=STGCN_FPS)
            if det.update(pf) is not None:
                fired = 1
                break
        y_pred.append(fired)
    return _metrics("Fall", y.tolist(), y_pred)


def evaluate_running(test_data: dict) -> dict:
    X, y = test_data["X"], test_data["y_running"]
    y_pred = []
    for i in tqdm(range(len(X)), desc="  Running eval", unit="win"):
        det = RunningDetector(fps=STGCN_FPS)
        fired = 0
        for t in range(len(X[i])):
            pf = pose_from_npy(X[i][t], frame_idx=t, fps=STGCN_FPS)
            if det.update(pf) is not None:
                fired = 1
                break
        y_pred.append(fired)
    return _metrics("Running", y.tolist(), y_pred)


def evaluate_inactivity(test_data: dict) -> dict:
    sequences = group_into_sequences(test_data, "inactivity")
    y_true, y_pred = [], []
    for seq_info in tqdm(sequences, desc="  Inactivity eval", unit="seq"):
        det = InactivityDetector(fps=STGCN_FPS, eval_mode=True)
        fired = 0
        for t in range(len(seq_info["X_seq"])):
            pf = pose_from_npy(seq_info["X_seq"][t], frame_idx=t, fps=STGCN_FPS)
            if det.update(pf) is not None:
                fired = 1
                break
        y_true.append(seq_info["label"])
        y_pred.append(fired)
    return _metrics("Inactivity", y_true, y_pred)


# =============================================================================
# Metrics printer
# =============================================================================

def _metrics(name: str, y_true: list, y_pred: list) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n{'='*52}")
    print(f"  {name.upper()}")
    print(f"{'='*52}")
    print(f"  Accuracy : {acc*100:6.2f}%")
    print(f"  Precision: {prec*100:6.2f}%")
    print(f"  Recall   : {rec*100:6.2f}%")
    print(f"  F1-score : {f1*100:6.2f}%")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    status = "[OK] >= 90% target met" if acc >= 0.90 else "[!!] Below 90% - tune thresholds"
    print(f"  {status}")

    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "cm": cm}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Auto-calibrate thresholds on train split (fast, vectorized)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Worker Behavior Detection - Rule-based Evaluation")
    print("  (pre-extracted X.npy, no video re-processing)")
    print("=" * 60)

    data = load_npy_dataset()
    train_data, test_data = train_test_split(data)

    if args.tune:
        print("\n--- Extracting features for tuning (train split) ---")
        feats_train = extract_window_features(train_data["X"])
        print("Feature extraction done.")

        print("\n--- Grid-search thresholds ---")
        best_fall = tune_fall(feats_train, train_data["y_fall"])
        FALL["angle_threshold"]      = best_fall["angle"]
        FALL["hip_height_threshold"] = best_fall["hip"]

        best_run = tune_running(feats_train, train_data["y_running"])
        RUNNING["step_freq_threshold"]        = best_run["freq"]
        RUNNING["horizontal_speed_threshold"] = best_run["speed"]

        best_inact = tune_inactivity(feats_train, train_data["y_inactivity"])
        INACTIVITY["movement_threshold"] = best_inact["mvt"]

        print("\n--- Tuned thresholds (update src/config.py with these) ---")
        print(f"  FALL      angle_threshold       = {best_fall['angle']:.1f}")
        print(f"  FALL      hip_height_threshold  = {best_fall['hip']:.3f}")
        print(f"  RUNNING   step_freq_threshold   = {best_run['freq']:.2f}")
        print(f"  RUNNING   horizontal_speed_thr  = {best_run['speed']:.4f}")
        print(f"  INACTIVITY movement_threshold   = {best_inact['mvt']:.5f}")

    print(f"\n--- Test evaluation ({len(test_data['X'])} windows) ---")
    results = {}
    results["fall"]       = evaluate_fall(test_data)
    results["running"]    = evaluate_running(test_data)
    results["inactivity"] = evaluate_inactivity(test_data)

    # Summary table
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*52}")
    for r in results.values():
        print(f"  {r['name']:<14} {r['accuracy']*100:>9.2f}%"
              f" {r['precision']*100:>9.2f}%"
              f" {r['recall']*100:>7.2f}%"
              f" {r['f1']*100:>7.2f}%")

    mean_acc = np.mean([r["accuracy"] for r in results.values()])
    print(f"\n  Mean accuracy  : {mean_acc*100:.2f}%")

    if all(r["accuracy"] >= 0.90 for r in results.values()):
        print("  ALL tasks >= 90% - target achieved!")
    else:
        below = [r["name"] for r in results.values() if r["accuracy"] < 0.90]
        print(f"  Tasks below 90%: {below}")
        print("  -> Re-run with --tune and update src/config.py")


if __name__ == "__main__":
    main()
