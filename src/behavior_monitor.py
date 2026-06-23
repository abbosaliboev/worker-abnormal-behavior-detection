"""
BehaviorMonitor — multi-person, tracking-aware orchestrator.

Uses YOLO ByteTracker to assign persistent IDs to each worker.
Each person gets independent fall, running, and inactivity detectors.

Pipeline per frame:
  YOLO.track() -> [PoseFrame_#1, PoseFrame_#2, ...]
      -> FallDetector (per person)
      -> RunningDetector (per person)
      -> InactivityDetector (per person, with spatial position tracking)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from src.pose_extractor import PoseExtractor, PoseFrame, draw_all_poses
from fall_detection.detector import FallDetector, FallEvent
from running_detection.detector import RunningDetector, RunEvent
from inactivity_detection.detector import InactivityDetector, InactivityEvent
from src.config import STGCN_FPS


@dataclass
class BehaviorAlert:
    alert_type:    str     # "FALL" | "RUNNING" | "INACTIVITY"
    track_id:      int
    frame_idx:     int
    timestamp_sec: float
    message:       str
    confidence:    float
    raw_event:     object


class _PersonDetectors:
    """Bundle of detectors for one tracked person."""
    def __init__(self, fps: float, eval_mode: bool):
        self.fall    = FallDetector(fps=fps)
        self.running = RunningDetector(fps=fps)
        # Inactivity timer is shared (multi-person aware)


class BehaviorMonitor:
    """
    Multi-person behavior monitor with YOLO tracking.

    Usage:
        monitor = BehaviorMonitor()
        with monitor:
            for frame in camera:
                alerts = monitor.update(frame)
    """

    def __init__(self, fps: float = STGCN_FPS,
                 eval_mode: bool = False,
                 use_tracking: bool = True):
        self.fps          = fps
        self._pose_ext    = PoseExtractor(use_tracking=use_tracking)
        self._inact       = InactivityDetector(fps=fps, eval_mode=eval_mode)
        self._persons: dict[int, _PersonDetectors] = {}
        self._frame_idx   = 0
        self._alerts: list[BehaviorAlert] = []
        self._last_poses:  list[PoseFrame] = []

    # ── Frame update ──────────────────────────────────────────────────────────

    def update(self, bgr_frame: np.ndarray) -> list[BehaviorAlert]:
        """Extract poses with tracking and run all detectors."""
        poses = self._pose_ext.process_frame(bgr_frame, self._frame_idx, self.fps)
        self._last_poses = poses
        alerts = self._process_poses(poses)
        self._frame_idx += 1
        return alerts

    def update_pose(self, pose_frame: PoseFrame) -> list[BehaviorAlert]:
        """Single-person backwards-compatible interface."""
        self._last_poses = [pose_frame]
        alerts = self._process_poses([pose_frame])
        self._frame_idx = pose_frame.frame_idx
        return alerts

    def _process_poses(self, poses: list[PoseFrame]) -> list[BehaviorAlert]:
        new_alerts: list[BehaviorAlert] = []

        # Inactivity: multi-person update
        inact_events = self._inact.update_all(poses)
        for ie in inact_events:
            a = BehaviorAlert(
                alert_type="INACTIVITY",
                track_id=ie.track_id,
                frame_idx=ie.frame_idx,
                timestamp_sec=ie.timestamp_sec,
                message=(f"[W#{ie.track_id}] INACTIVITY "
                         f"still={ie.duration_sec:.0f}s"),
                confidence=min(1.0, ie.duration_sec / 300.0),
                raw_event=ie,
            )
            new_alerts.append(a)
            self._alerts.append(a)

        # Fall + Running: per-person
        for pose in poses:
            if not pose.valid:
                continue
            tid = pose.track_id

            if tid not in self._persons:
                self._persons[tid] = _PersonDetectors(
                    fps=self.fps,
                    eval_mode=False,
                )
            det = self._persons[tid]

            fe: Optional[FallEvent] = det.fall.update(pose)
            if fe:
                a = BehaviorAlert(
                    alert_type="FALL",
                    track_id=tid,
                    frame_idx=fe.frame_idx,
                    timestamp_sec=fe.timestamp_sec,
                    message=(f"[W#{tid}] FALL DETECTED "
                             f"angle={fe.body_angle:.1f} "
                             f"rate={fe.angle_rate:.0f}deg/s"),
                    confidence=fe.confidence,
                    raw_event=fe,
                )
                new_alerts.append(a)
                self._alerts.append(a)
                # Reset this person's inactivity when they fall
                self._inact.reset(tid)

            re: Optional[RunEvent] = det.running.update(pose)
            if re:
                a = BehaviorAlert(
                    alert_type="RUNNING",
                    track_id=tid,
                    frame_idx=re.frame_idx,
                    timestamp_sec=re.timestamp_sec,
                    message=(f"[W#{tid}] UNSAFE RUNNING "
                             f"freq={re.step_frequency:.1f}Hz"),
                    confidence=re.score,
                    raw_event=re,
                )
                new_alerts.append(a)
                self._alerts.append(a)

        return new_alerts

    # ── Annotated frame ───────────────────────────────────────────────────────

    def annotate(self, bgr_frame: np.ndarray,
                 alerts: Optional[list[BehaviorAlert]] = None) -> np.ndarray:
        out = draw_all_poses(bgr_frame, self._last_poses)
        h, w = out.shape[:2]
        COLORS = {"FALL": (0, 0, 255),
                  "RUNNING": (0, 140, 255),
                  "INACTIVITY": (0, 165, 255)}

        # Status bar — show all tracked persons' still durations
        cv2.rectangle(out, (0, 0), (w, 26), (30, 30, 30), -1)
        durations = self._inact.all_still_durations()
        txt = "  ".join(f"W#{tid}:{d:.0f}s" for tid, d in durations.items())
        cv2.putText(out, txt or "No persons tracked",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # Recent alerts
        recent = (alerts or self._alerts)[-4:]
        for i, al in enumerate(reversed(recent)):
            color = COLORS.get(al.alert_type, (255, 255, 255))
            y = h - 12 - i * 24
            cv2.rectangle(out, (0, y - 16), (w, y + 6), (0, 0, 0), -1)
            cv2.putText(out, al.message, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
        return out

    # ── Housekeeping ──────────────────────────────────────────────────────────

    @property
    def all_alerts(self) -> list[BehaviorAlert]:
        return list(self._alerts)

    def reset(self):
        self._persons.clear()
        self._inact.reset()
        self._alerts.clear()
        self._frame_idx = 0

    def close(self):
        self._pose_ext.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
