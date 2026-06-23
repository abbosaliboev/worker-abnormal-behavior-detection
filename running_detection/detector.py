"""
Rule-based Running / Unsafe-Speed Detector.

A person is classified as RUNNING when (score-based voting):
  • step_frequency          >= RUNNING["step_freq_threshold"]          (2+ Hz)
  • horizontal_speed        >= RUNNING["horizontal_speed_threshold"]
  • vertical_oscillation    >= RUNNING["vertical_oscillation_threshold"]
  • knee_angle_variance     >= RUNNING["knee_angle_variance_threshold"]

At least 3 of these 4 conditions must be true, AND the composite score
must exceed 0.55, for the "running" label to fire.

Note: UP-Fall dataset Activity 8 ("running") is the positive class here.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.config import RUNNING, STGCN_FPS
from src.feature_extractor import FeatureBuffer
from src.pose_extractor import PoseFrame


class RunState(Enum):
    STILL     = auto()
    WALKING   = auto()
    RUNNING   = auto()
    COOLDOWN  = auto()


@dataclass
class RunEvent:
    frame_idx: int
    timestamp_sec: float
    step_frequency: float
    horizontal_speed: float
    vertical_oscillation: float
    knee_angle_variance: float
    score: float   # 0-1 composite confidence


class RunningDetector:
    def __init__(self, fps: float = STGCN_FPS):
        self.fps = fps
        self._buf = FeatureBuffer(window_frames=RUNNING["window_frames"], fps=fps)
        self._state = RunState.STILL
        self._run_frame_count = 0
        self._cooldown_frames_left = 0
        self._last_event: Optional[RunEvent] = None

    def update(self, pose_frame: PoseFrame) -> Optional[RunEvent]:
        """Feed one frame. Returns RunEvent when running is first confirmed."""
        self._buf.push(pose_frame)

        if self._state == RunState.COOLDOWN:
            self._cooldown_frames_left -= 1
            if self._cooldown_frames_left <= 0:
                self._state = RunState.STILL
            return None

        score, conditions = self._evaluate()

        if conditions >= 3 and score >= 0.55:
            self._run_frame_count += 1
            self._state = RunState.RUNNING
        else:
            self._run_frame_count = 0
            self._state = RunState.STILL if score < 0.25 else RunState.WALKING
            return None

        if self._run_frame_count == RUNNING["min_run_frames"]:
            event = self._make_event(score)
            self._last_event = event
            self._cooldown_frames_left = int(RUNNING["cooldown_seconds"] * self.fps)
            self._state = RunState.COOLDOWN
            self._run_frame_count = 0
            return event

        return None

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def last_event(self) -> Optional[RunEvent]:
        return self._last_event

    def reset(self):
        self._buf = FeatureBuffer(window_frames=RUNNING["window_frames"], fps=self.fps)
        self._state = RunState.STILL
        self._run_frame_count = 0
        self._cooldown_frames_left = 0
        self._last_event = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, int]:
        """Returns (composite_score 0-1, number_of_conditions_met)."""
        freq  = self._buf.step_frequency()
        speed = self._buf.com_horizontal_speed()
        osc   = self._buf.com_vertical_oscillation()
        kvar  = self._buf.knee_angle_variance()

        thr_f = RUNNING["step_freq_threshold"]
        thr_s = RUNNING["horizontal_speed_threshold"]
        thr_o = RUNNING["vertical_oscillation_threshold"]
        thr_k = RUNNING["knee_angle_variance_threshold"]

        # Normalised scores (capped at 1)
        s_freq  = min(1.0, freq  / thr_f)  if thr_f  > 0 else 0.0
        s_speed = min(1.0, speed / thr_s)  if thr_s  > 0 else 0.0
        s_osc   = min(1.0, osc   / thr_o)  if thr_o  > 0 else 0.0
        s_kvar  = min(1.0, kvar  / thr_k)  if thr_k  > 0 else 0.0

        # Weighted composite (step_freq and speed are most reliable)
        composite = 0.35 * s_freq + 0.35 * s_speed + 0.15 * s_osc + 0.15 * s_kvar

        conditions_met = sum([
            freq  >= thr_f,
            speed >= thr_s,
            osc   >= thr_o,
            kvar  >= thr_k,
        ])
        return composite, conditions_met

    def _make_event(self, score: float) -> RunEvent:
        current = self._buf.current()
        frame_idx = current["frame_idx"] if current else 0
        ts = current["timestamp"] if current else 0.0
        return RunEvent(
            frame_idx=frame_idx,
            timestamp_sec=ts,
            step_frequency=self._buf.step_frequency(),
            horizontal_speed=self._buf.com_horizontal_speed(),
            vertical_oscillation=self._buf.com_vertical_oscillation(),
            knee_angle_variance=self._buf.knee_angle_variance(),
            score=score,
        )
