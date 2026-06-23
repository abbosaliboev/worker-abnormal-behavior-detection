"""
Rule-based Fall Detector.

Three complementary rule sets (majority / confirmation logic):

  A) Angle-rate gate (PRIMARY discriminator):
     • body_angle change rate > FALL["angle_rate_threshold"]  (deg/sec)
     • This separates RAPID falls (70-140°/sec) from SLOW lying-down (2-5°/sec)
     • Also separates from walking (8-12°/sec) and standing (0-3°/sec)

  B) Geometry confirmation:
     • body_tilt_angle > FALL["angle_threshold"]  (spine ≥ 70° from vertical)
     • aspect_ratio    < FALL["aspect_ratio_threshold"] (bbox is wide/horizontal)

  C) Kinematics (Butterworth-filtered hip velocity, secondary):
     • downward_velocity > FALL["vel_threshold"]

A fall is confirmed when:
  (A) angle_rate gate fires  AND  (B) geometry confirms
  OR
  (C) kinematics fires  AND  (B) geometry confirms

Requires FALL["min_fall_frames"] consecutive triggered frames.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt

from src.config import FALL, STGCN_FPS
from src.feature_extractor import FeatureBuffer, body_tilt_angle, hip_height_normalized, pose_aspect_ratio
from src.pose_extractor import PoseFrame


class FallState(Enum):
    UPRIGHT  = auto()
    FALLING  = auto()   # triggered, not yet confirmed
    FALLEN   = auto()   # confirmed
    COOLDOWN = auto()


@dataclass
class FallEvent:
    frame_idx:    int
    timestamp_sec: float
    body_angle:   float
    aspect_ratio: float
    angle_rate:   float     # deg/sec
    trigger:      str       # "angle_rate" | "kinematics" | "both"
    confidence:   float


class FallDetector:
    """
    Rule-based fall detector using YOLO 17-joint pose data.
    Primary discriminator: angle rate (fall speed) — separates falls from lying-down.
    """

    def __init__(self, fps: float = STGCN_FPS):
        self.fps = fps
        self._buf        = FeatureBuffer(window_frames=30, fps=fps)
        self._angle_buf: deque[float] = deque(maxlen=FALL["angle_rate_window"] + 2)
        self._hip_y_buf: deque[float] = deque(maxlen=30)
        self._state         = FallState.UPRIGHT
        self._trigger_count = 0
        self._cooldown_left = 0
        self._last_event: Optional[FallEvent] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, pose_frame: PoseFrame) -> Optional[FallEvent]:
        feat = self._buf.push(pose_frame)

        if self._state == FallState.COOLDOWN:
            self._cooldown_left -= 1
            if self._cooldown_left <= 0:
                self._state = FallState.UPRIGHT
            return None

        if feat is None:
            self._trigger_count = 0
            return None

        self._angle_buf.append(feat["body_tilt_angle"])
        self._hip_y_buf.append(feat["hip_height"])

        angle_rate_trigger = self._angle_rate_trigger(feat)
        kine_trigger       = self._kinematics_trigger()
        geo_ok             = self._geometry_ok(feat)

        triggered = (angle_rate_trigger or kine_trigger) and geo_ok

        if triggered:
            self._trigger_count += 1
            self._state = FallState.FALLING
        else:
            self._trigger_count = 0
            self._state = FallState.UPRIGHT
            return None

        if self._trigger_count >= FALL["min_fall_frames"]:
            trigger_label = (
                "both"        if angle_rate_trigger and kine_trigger else
                "angle_rate"  if angle_rate_trigger else
                "kinematics"
            )
            event = self._make_event(feat, trigger_label)
            self._last_event = event
            self._cooldown_left = int(FALL["cooldown_seconds"] * self.fps)
            self._state = FallState.COOLDOWN
            self._trigger_count = 0
            return event

        return None

    @property
    def state(self) -> FallState:
        return self._state

    @property
    def last_event(self) -> Optional[FallEvent]:
        return self._last_event

    def reset(self):
        self._buf         = FeatureBuffer(window_frames=30, fps=self.fps)
        self._angle_buf   = deque(maxlen=FALL["angle_rate_window"] + 2)
        self._hip_y_buf   = deque(maxlen=30)
        self._state       = FallState.UPRIGHT
        self._trigger_count = 0
        self._cooldown_left = 0
        self._last_event  = None

    # ── Rule A: angle rate (PRIMARY) ──────────────────────────────────────────

    def _angle_rate_trigger(self, feat: dict) -> bool:
        """
        True if body angle is changing faster than threshold (deg/sec).
        Computed as change over last FALL["angle_rate_window"] frames.
        Falls: 70-140 deg/sec. Lying down: 2-5 deg/sec.
        """
        window = FALL["angle_rate_window"]
        if len(self._angle_buf) < window + 1:
            return False
        past_angle    = list(self._angle_buf)[-window - 1]
        current_angle = feat["body_tilt_angle"]
        delta_deg     = abs(current_angle - past_angle)
        rate_deg_sec  = delta_deg / (window / self.fps)
        return rate_deg_sec > FALL["angle_rate_threshold"]

    # ── Rule B: geometry confirmation ─────────────────────────────────────────

    def _geometry_ok(self, feat: dict) -> bool:
        """Person must be tilted AND (horizontal bbox OR low in frame)."""
        angle = feat["body_tilt_angle"]
        ar    = feat["aspect_ratio"]
        hip_h = feat["hip_height"]
        return (angle > FALL["angle_threshold"] and
                (ar < FALL["aspect_ratio_threshold"] or hip_h > 0.58))

    # ── Rule C: kinematics (Butterworth hip velocity) ─────────────────────────

    def _kinematics_trigger(self) -> bool:
        data = np.array(self._hip_y_buf)
        if len(data) < 6:
            return False
        try:
            vel, _ = _hip_kinematics(data, self.fps,
                                     FALL["pos_cutoff_hz"],
                                     FALL["vel_cutoff_hz"])
        except Exception:
            return False
        return vel > FALL["vel_threshold"]

    # ── Event builder ─────────────────────────────────────────────────────────

    def _make_event(self, feat: dict, trigger: str) -> FallEvent:
        angle = feat["body_tilt_angle"]
        ar    = feat["aspect_ratio"]
        window = FALL["angle_rate_window"]
        past  = list(self._angle_buf)[-window - 1] if len(self._angle_buf) > window else 0.0
        rate  = abs(angle - past) / (window / self.fps)

        conf = min(1.0, rate / FALL["angle_rate_threshold"]) * 0.6 + \
               min(1.0, angle / FALL["angle_threshold"]) * 0.4

        return FallEvent(
            frame_idx=feat["frame_idx"],
            timestamp_sec=feat["timestamp"],
            body_angle=angle,
            aspect_ratio=ar,
            angle_rate=rate,
            trigger=trigger,
            confidence=float(conf),
        )


# ─── Butterworth kinematics ───────────────────────────────────────────────────

def _lowpass(data: np.ndarray, fc: float, fps: float, order: int = 2) -> np.ndarray:
    nyq = fps / 2.0
    wn  = min(fc / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    padlen = min(3 * (max(len(a), len(b)) - 1), len(data) - 1)
    if padlen < 1:
        return data.copy()
    return filtfilt(b, a, data, padlen=padlen)


def _hip_kinematics(hip_y: np.ndarray, fps: float,
                    pos_fc: float, vel_fc: float) -> tuple[float, float]:
    t      = np.arange(len(hip_y)) / fps
    hip_f  = _lowpass(hip_y, pos_fc, fps)
    vel    = np.gradient(hip_f, t)
    vel_f  = _lowpass(vel, vel_fc, fps)
    acc_f  = np.gradient(vel_f, t)
    return float(vel_f.max()), float(np.abs(acc_f).max())
