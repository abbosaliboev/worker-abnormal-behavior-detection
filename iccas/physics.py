"""
Stage-2 Physics Rule Filter.

Takes a skeleton sequence (T, V, C) and computes hip kinematics:
  position -> 4 Hz Butterworth -> velocity -> 8 Hz Butterworth -> acceleration

Decision rule:
  fall confirmed  if  max_velocity > vel_threshold
                  AND max_abs_acc  > acc_threshold

Thresholds are fitted from training data (lowest `percentile`-th percentile
of fall sequences, so the filter remains permissive for true falls).

MediaPipe joint indices used:
  LEFT_HIP  = 23  (y-coordinate index 1; 0=top, 1=bottom -> downward = +)
  RIGHT_HIP = 24
"""

import numpy as np
from scipy.signal import butter, filtfilt

LEFT_HIP  = 11      # COCO keypoint index (YOLO-pose)
RIGHT_HIP = 12
Y_COORD   = 1       # normalized image y (0=top, 1=bottom)


class PhysicsFilter:
    """
    Args:
        fps          : frame rate of the skeleton sequence (default 19 Hz)
        pos_fc       : position low-pass cutoff (Hz)
        vel_fc       : velocity low-pass cutoff (Hz)
        acc_fc       : acceleration low-pass cutoff (Hz)
        vel_threshold: downward velocity threshold (normalized units/s)
        acc_threshold: absolute acceleration threshold (normalized units/s²)
    """

    def __init__(
        self,
        fps: float = 19.0,
        pos_fc: float = 4.0,
        vel_fc: float = 8.0,
        acc_fc: float = 6.0,
        vel_threshold: float = None,
        acc_threshold: float = None,
    ):
        self.fps = fps
        self.pos_fc = pos_fc
        self.vel_fc = vel_fc
        self.acc_fc = acc_fc
        self.vel_threshold = vel_threshold
        self.acc_threshold = acc_threshold

    # ── internal ─────────────────────────────────────────────────────────────

    def _lowpass(self, data: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
        nyq = self.fps / 2.0
        wn  = min(fc / nyq, 0.99)
        b, a = butter(order, wn, btype="low")
        # filtfilt padlen must be < signal length
        padlen = min(3 * (max(len(a), len(b)) - 1), len(data) - 1)
        if padlen < 1:
            return data.copy()
        return filtfilt(b, a, data, padlen=padlen)

    # ── public API ────────────────────────────────────────────────────────────

    def extract_features(self, seq: np.ndarray) -> dict:
        """
        seq : (T, V, C) skeleton sequence (MediaPipe normalized coords)
        Returns a dict with scalar physics features.
        """
        T = seq.shape[0]
        t = np.arange(T) / self.fps

        # mid-hip Y  (downward positive)
        hip_y = (seq[:, LEFT_HIP, Y_COORD] + seq[:, RIGHT_HIP, Y_COORD]) / 2.0

        # filter position
        hip_y_f = self._lowpass(hip_y, self.pos_fc)

        # velocity (downward positive)
        vel   = np.gradient(hip_y_f, t)
        vel_f = self._lowpass(vel, self.vel_fc)

        # acceleration
        acc   = np.gradient(vel_f, t)
        acc_f = self._lowpass(acc, self.acc_fc)

        return {
            "max_velocity"  : float(vel_f.max()),
            "max_abs_acc"   : float(np.abs(acc_f).max()),
            "hip_drop"      : float(hip_y_f.max() - hip_y_f.min()),
            "hip_y_filtered": hip_y_f,
            "velocity"      : vel_f,
            "acceleration"  : acc_f,
        }

    def predict(self, seq: np.ndarray) -> int:
        """
        Returns 1 (fall confirmed) or 0.
        Thresholds must be set (via fit() or manually).
        """
        if self.vel_threshold is None or self.acc_threshold is None:
            raise RuntimeError("Call fit() or set vel_threshold / acc_threshold first.")
        f = self.extract_features(seq)
        return int(
            f["max_velocity"] > self.vel_threshold
            and f["max_abs_acc"] > self.acc_threshold
        )

    def fit(self, X: np.ndarray, y: np.ndarray, percentile: float = 20.0):
        """
        Derive thresholds from training data.

        Uses the `percentile`-th percentile of FALL sequences so that
        (100-percentile)% of true falls pass Stage 2 (permissive).

        Args:
            X          : (N, T, V, C)
            y          : (N,)  1=fall, 0=no-fall
            percentile : lower bound percentile on fall feature distribution
        """
        fall_vels, fall_accs = [], []
        for i in range(len(y)):
            if y[i] == 1:
                f = self.extract_features(X[i])
                fall_vels.append(f["max_velocity"])
                fall_accs.append(f["max_abs_acc"])

        self.vel_threshold = float(np.percentile(fall_vels, percentile))
        self.acc_threshold = float(np.percentile(fall_accs, percentile))
        print(
            f"[PhysicsFilter] fitted thresholds — "
            f"vel > {self.vel_threshold:.5f}  "
            f"acc > {self.acc_threshold:.5f}  "
            f"(from {len(fall_vels)} fall sequences, p{percentile})"
        )

    def search_thresholds(self, X: np.ndarray, y: np.ndarray, stage1_preds: np.ndarray):
        """
        Grid-search over (vel_threshold, acc_threshold) to maximise fall-F1
        on a validation set, given Stage-1 predictions.

        Only sequences where stage1_preds == 1 are sent to Stage 2.
        A correct fall requires both stages to agree.

        Returns (best_vel, best_acc, best_f1).
        """
        from sklearn.metrics import f1_score

        # collect physics features for all sequences
        feats = [self.extract_features(X[i]) for i in range(len(X))]
        vel_vals = np.array([f["max_velocity"] for f in feats])
        acc_vals = np.array([f["max_abs_acc"]  for f in feats])

        vel_candidates = np.percentile(vel_vals[y == 1], np.arange(0, 100, 5))
        acc_candidates = np.percentile(acc_vals[y == 1], np.arange(0, 100, 5))

        best_f1, best_vel, best_acc = -1, 0.0, 0.0
        for vt in vel_candidates:
            for at in acc_candidates:
                physics_pass = (vel_vals > vt) & (acc_vals > at)
                preds = np.where(stage1_preds == 1, physics_pass.astype(int), 0)
                f1 = f1_score(y, preds, pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_vel = vt
                    best_acc = at

        self.vel_threshold = float(best_vel)
        self.acc_threshold = float(best_acc)
        print(
            f"[PhysicsFilter] grid-search — "
            f"vel > {self.vel_threshold:.5f}  "
            f"acc > {self.acc_threshold:.5f}  "
            f"best_val_fall_F1 = {best_f1:.4f}"
        )
        return best_vel, best_acc, best_f1
