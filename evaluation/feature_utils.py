"""
Vectorized feature extraction from pre-extracted X.npy keypoints.
Used internally by evaluation scripts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import KP, KP_VISIBILITY_THRESHOLD


def _angle_deg(v1, v2):
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos = np.einsum("...i,...i->...", v1, v2) / (n1[..., 0] * n2[..., 0] + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _joint_angle_batch(X, a, b, c):
    ba = X[:, :, a, :2] - X[:, :, b, :2]
    bc = X[:, :, c, :2] - X[:, :, b, :2]
    return _angle_deg(ba, bc)


def extract_window_features(X: np.ndarray, fps: float = 19.0) -> dict:
    """
    X: (N, T, V, C)  — N windows, T frames, 17 joints, (x, y, conf)
    Returns dict of (N,) arrays with per-window aggregate features.
    """
    N, T, V, C = X.shape
    dt = 1.0 / fps

    L_SH, R_SH = KP["l_shoulder"], KP["r_shoulder"]
    L_HIP, R_HIP = KP["l_hip"], KP["r_hip"]
    L_KN, R_KN = KP["l_knee"], KP["r_knee"]
    L_AN, R_AN = KP["l_ankle"], KP["r_ankle"]

    xy = X[..., :2]
    cf = X[..., 2]

    mid_hip = (xy[:, :, L_HIP] + xy[:, :, R_HIP]) / 2.0
    mid_sh  = (xy[:, :, L_SH]  + xy[:, :, R_SH])  / 2.0

    # Body tilt angle
    spine    = mid_sh - mid_hip
    vertical = np.array([0.0, -1.0])
    body_angle = _angle_deg(spine, np.broadcast_to(vertical, spine.shape))

    # Hip Y
    hip_y = mid_hip[:, :, 1]

    # Aspect ratio
    vis_mask = cf >= KP_VISIBILITY_THRESHOLD
    x_vals = np.where(vis_mask, xy[..., 0], np.nan)
    y_vals = np.where(vis_mask, xy[..., 1], np.nan)
    bbox_h = np.nanmax(y_vals, axis=2) - np.nanmin(y_vals, axis=2)
    bbox_w = np.nanmax(x_vals, axis=2) - np.nanmin(x_vals, axis=2)
    aspect_ratio = np.where(bbox_w > 1e-4, bbox_h / (bbox_w + 1e-8), 5.0)

    # Hip velocity
    hip_vel = np.gradient(hip_y, dt, axis=1)

    # CoM
    CoM = (mid_hip + mid_sh) / 2.0
    com_x = CoM[:, :, 0]
    com_y = CoM[:, :, 1]
    com_dx = np.abs(np.diff(com_x, axis=1))

    # Knee angles
    l_knee = _joint_angle_batch(X, L_HIP, L_KN, L_AN)
    r_knee = _joint_angle_batch(X, R_HIP, R_KN, R_AN)
    knee_diff = l_knee - r_knee
    sign_changes = np.diff(np.sign(knee_diff), axis=1)
    zc_count = (sign_changes != 0).sum(axis=1)
    step_freq = zc_count / (2.0 * T / fps)
    knee_var = knee_diff.var(axis=1)

    # Keypoint displacement
    kp_disp = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    vis_both = vis_mask[:, :-1] & vis_mask[:, 1:]
    kp_disp_masked = np.where(vis_both, kp_disp, np.nan)
    mean_disp = np.nanmean(kp_disp_masked, axis=(1, 2))
    mean_disp = np.where(np.isnan(mean_disp), 0.0, mean_disp)

    return {
        "max_body_angle":   body_angle.max(axis=1),
        "max_hip_y":        hip_y.max(axis=1),
        "min_aspect_ratio": aspect_ratio.min(axis=1),
        "max_hip_vel":      hip_vel.max(axis=1),
        "step_freq":        step_freq,
        "horiz_speed":      com_dx.mean(axis=1),
        "vert_osc":         com_y.std(axis=1),
        "knee_var":         knee_var,
        "mean_kp_disp":     mean_disp,
    }
