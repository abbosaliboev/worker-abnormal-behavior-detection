"""
Tracking-based Inactivity Detector.

Each tracked person (by YOLO track_id) gets an independent inactivity timer.
A person is INACTIVE when their SPATIAL POSITION (CoM x,y) does not change
significantly for >= inactivity_timeout_seconds.

Using position tracking (not just keypoint displacement) is more robust:
  - A person fidgeting in place → position barely changes → still inactive
  - A person walking → position changes → not inactive

AND-rule on 3 features (per frame):
  1. CoM spatial displacement  < pos_threshold   (not moving through space)
  2. Keypoint displacement      < kp_threshold    (limbs mostly still)
  3. Body angle stability       < angle_threshold  (stable posture, not jumping)

Alert fires when all three hold for >= timeout_seconds for a given track_id.

In cascade architecture:
  - Running detector fires first  (person running → not inactive)
  - Fall detector fires first     (person fallen  → fall alert)
  - Inactivity timer only runs when neither of the above fired
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from collections import deque

import numpy as np

from src.config import INACTIVITY, STGCN_FPS, KP
from src.feature_extractor import FeatureBuffer, body_tilt_angle, center_of_mass_xy
from src.pose_extractor import PoseFrame


class InactivityState(Enum):
    ACTIVE   = auto()
    INACTIVE = auto()


@dataclass
class InactivityEvent:
    track_id:       int
    frame_idx:      int
    timestamp_sec:  float
    duration_sec:   float
    avg_speed:      float   # mean CoM displacement per frame during idle period


class _PersonTimer:
    """
    Per-person inactivity timer, keyed by track_id.
    Maintains a short history of CoM positions and keypoint features.
    """

    def __init__(self, fps: float, timeout: float,
                 pos_thr: float, kp_thr: float, angle_thr: float):
        self.fps       = fps
        self.timeout   = timeout
        self.pos_thr   = pos_thr
        self.kp_thr    = kp_thr
        self.angle_thr = angle_thr

        self._buf      = FeatureBuffer(window_frames=INACTIVITY["window_frames"], fps=fps)
        self._com_hist: deque[np.ndarray] = deque(maxlen=int(fps * 10))  # 10s CoM trail
        self._ang_hist: deque[float]      = deque(maxlen=int(fps * 10))

        self.state        = InactivityState.ACTIVE
        self.still_frames = 0
        self._total_speed = 0.0
        self.last_pos: Optional[np.ndarray] = None

    def update(self, pose_frame: PoseFrame) -> Optional[float]:
        """
        Feed one frame. Returns duration_sec if newly inactive, else None.
        """
        feat = self._buf.push(pose_frame)
        if feat is None:
            return None

        kp      = feat["kps"]
        com_now = center_of_mass_xy(kp)
        angle   = feat["body_tilt_angle"]

        self._com_hist.append(com_now)
        self._ang_hist.append(angle)

        # Feature 1: spatial CoM displacement since last frame
        if self.last_pos is not None:
            pos_disp = float(np.linalg.norm(com_now - self.last_pos))
        else:
            pos_disp = 0.0
        self.last_pos = com_now.copy()

        # Feature 2: keypoint displacement (from buffer)
        kp_disp = self._buf.mean_keypoint_displacement()

        # Feature 3: body angle stability (std over recent history)
        angle_std = float(np.std(list(self._ang_hist))) if len(self._ang_hist) > 2 else 0.0

        # AND rule
        all_still = (pos_disp   < self.pos_thr  and
                     kp_disp    < self.kp_thr    and
                     angle_std  < self.angle_thr)

        if all_still:
            self.still_frames += 1
            self._total_speed += pos_disp
        else:
            self.still_frames  = 0
            self._total_speed  = 0.0
            self._com_hist.clear()
            self._ang_hist.clear()
            self.state = InactivityState.ACTIVE
            return None

        duration = self.still_frames / self.fps
        if self.state == InactivityState.ACTIVE and duration >= self.timeout:
            self.state = InactivityState.INACTIVE
            avg_speed = self._total_speed / max(self.still_frames, 1)
            return duration
        return None

    def reset(self):
        self._buf       = FeatureBuffer(window_frames=INACTIVITY["window_frames"], fps=self.fps)
        self._com_hist  = deque(maxlen=int(self.fps * 10))
        self._ang_hist  = deque(maxlen=int(self.fps * 10))
        self.state      = InactivityState.ACTIVE
        self.still_frames = 0
        self._total_speed = 0.0
        self.last_pos   = None

    @property
    def still_duration_sec(self) -> float:
        return self.still_frames / self.fps


class InactivityDetector:
    """
    Multi-person, tracking-based inactivity detector.

    Each unique track_id gets its own independent timer.
    Falls back to single-person mode (track_id=-1) when tracking is disabled.

    Args:
        fps       : stream FPS
        eval_mode : use shorter eval_timeout
    """

    # Thresholds
    POS_THR   = 0.008   # CoM spatial displacement per frame (normalized)
    KP_THR    = INACTIVITY["kp_disp_threshold"]
    ANGLE_THR = INACTIVITY["body_angle_std_threshold"]

    def __init__(self, fps: float = STGCN_FPS, eval_mode: bool = False):
        self.fps      = fps
        self._timeout = (INACTIVITY["eval_timeout_seconds"]
                         if eval_mode
                         else INACTIVITY["inactivity_timeout_seconds"])
        self._timers: dict[int, _PersonTimer] = {}
        self._last_events: list[InactivityEvent] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, pose_frame: PoseFrame) -> Optional[InactivityEvent]:
        """
        Feed one PoseFrame (single person).
        Use update_all() for multi-person tracking.
        """
        events = self.update_all([pose_frame])
        return events[0] if events else None

    def update_all(self, poses: list[PoseFrame]) -> list[InactivityEvent]:
        """
        Feed all detected persons for this frame.
        Returns list of new InactivityEvents (one per newly inactive person).
        """
        events = []
        seen_ids = set()

        for pose in poses:
            if not pose.valid:
                continue
            tid = pose.track_id if pose.track_id >= 0 else -1
            seen_ids.add(tid)

            if tid not in self._timers:
                self._timers[tid] = _PersonTimer(
                    fps=self.fps, timeout=self._timeout,
                    pos_thr=self.POS_THR,
                    kp_thr=self.KP_THR,
                    angle_thr=self.ANGLE_THR,
                )

            duration = self._timers[tid].update(pose)
            if duration is not None:
                evt = InactivityEvent(
                    track_id=tid,
                    frame_idx=pose.frame_idx,
                    timestamp_sec=pose.timestamp_sec,
                    duration_sec=duration,
                    avg_speed=0.0,
                )
                events.append(evt)
                self._last_events.append(evt)

        return events

    def still_duration_sec(self, track_id: int = -1) -> float:
        """How long has person track_id been still (seconds)?"""
        timer = self._timers.get(track_id)
        return timer.still_duration_sec if timer else 0.0

    def all_still_durations(self) -> dict[int, float]:
        return {tid: t.still_duration_sec for tid, t in self._timers.items()}

    @property
    def state(self):
        """For single-person backwards compatibility."""
        from src.inactivity_detector import InactivityState
        t = self._timers.get(-1) or (next(iter(self._timers.values()), None))
        return t.state if t else InactivityState.ACTIVE

    @property
    def last_event(self) -> Optional[InactivityEvent]:
        return self._last_events[-1] if self._last_events else None

    def reset(self, track_id: Optional[int] = None):
        """Reset one person (track_id) or all if None."""
        if track_id is None:
            self._timers.clear()
            self._last_events.clear()
        elif track_id in self._timers:
            self._timers[track_id].reset()

    # Backwards-compat alias
    @property
    def movement_threshold(self) -> float:
        return self.KP_THR
