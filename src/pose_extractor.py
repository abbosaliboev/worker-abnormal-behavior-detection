"""
YOLO11n-pose wrapper with optional tracking (ByteTracker).

Returns per-person PoseFrame objects including track_id.
Tracking mode assigns persistent IDs across frames — needed for
per-person inactivity detection in multi-worker scenes.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from src.config import YOLO_POSE_MODEL, YOLO_CONF, YOLO_VERBOSE, NUM_JOINTS


@dataclass
class PoseFrame:
    keypoints:     np.ndarray   # (17, 3): x_norm, y_norm, conf
    frame_idx:     int
    timestamp_sec: float
    valid:         bool         # False if no person detected
    track_id:      int = -1     # -1 = tracking disabled or lost


class PoseExtractor:
    """
    YOLO11n-pose wrapper.

    Args:
        model_path : path to yolo11n-pose.pt
        use_tracking: enable ByteTracker (persist=True) for multi-person ID
    """

    def __init__(self, model_path: str = YOLO_POSE_MODEL,
                 use_tracking: bool = True):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._model.overrides["verbose"] = YOLO_VERBOSE
        self._use_tracking = use_tracking

    def process_frame(self, bgr_frame: np.ndarray,
                      frame_idx: int = 0,
                      fps: float = 30.0) -> list["PoseFrame"]:
        """
        Process one frame.

        Returns a LIST of PoseFrame objects — one per detected person.
        In single-person mode (tracking disabled), list has at most 1 element.

        For backwards compatibility, callers expecting a single PoseFrame
        can use process_frame_single().
        """
        h, w = bgr_frame.shape[:2]

        if self._use_tracking:
            results = self._model.track(bgr_frame, persist=True,
                                        conf=YOLO_CONF, verbose=YOLO_VERBOSE)
        else:
            results = self._model(bgr_frame, conf=YOLO_CONF, verbose=YOLO_VERBOSE)

        r = results[0]
        poses = []

        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return poses

        ky   = r.keypoints
        n    = len(ky.xy)

        # Get track IDs if available
        track_ids = [-1] * n
        if self._use_tracking and r.boxes is not None and r.boxes.id is not None:
            tids = r.boxes.id.cpu().numpy().astype(int)
            track_ids = tids.tolist()

        for i in range(n):
            kp = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
            xy   = ky.xy[i].cpu().numpy()
            conf = ky.conf[i].cpu().numpy() if ky.conf is not None else np.ones(NUM_JOINTS)
            kp[:, 0] = xy[:, 0] / max(w, 1)
            kp[:, 1] = xy[:, 1] / max(h, 1)
            kp[:, 2] = conf
            valid = kp[:, :2].sum() > 0
            poses.append(PoseFrame(
                keypoints=kp,
                frame_idx=frame_idx,
                timestamp_sec=frame_idx / fps,
                valid=valid,
                track_id=track_ids[i],
            ))

        return poses

    def process_frame_single(self, bgr_frame: np.ndarray,
                             frame_idx: int = 0,
                             fps: float = 30.0) -> "PoseFrame":
        """
        Backwards-compatible single-person interface.
        Returns the person with highest keypoint confidence, or empty PoseFrame.
        """
        poses = self.process_frame(bgr_frame, frame_idx, fps)
        if not poses:
            return PoseFrame(
                keypoints=np.zeros((NUM_JOINTS, 3), dtype=np.float32),
                frame_idx=frame_idx,
                timestamp_sec=frame_idx / fps,
                valid=False,
                track_id=-1,
            )
        # Pick highest confidence person
        best = max(poses, key=lambda p: p.keypoints[:, 2].sum())
        return best

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def pose_from_npy(kp_17x3: np.ndarray, frame_idx: int = 0,
                  fps: float = 19.0) -> PoseFrame:
    """Build PoseFrame from pre-extracted (17,3) array (for offline evaluation)."""
    valid = kp_17x3[:, :2].sum() > 0
    return PoseFrame(
        keypoints=kp_17x3.astype(np.float32),
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / fps,
        valid=valid,
        track_id=-1,
    )


# ─── Drawing ─────────────────────────────────────────────────────────────────

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

TRACK_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 100, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
]


def draw_pose(bgr_frame: np.ndarray, pose_frame: PoseFrame,
              kp_color=(0, 255, 0), bone_color=(255, 200, 0)) -> np.ndarray:
    if not pose_frame.valid:
        return bgr_frame
    out = bgr_frame.copy()
    h, w = out.shape[:2]

    # Use different color per track ID
    if pose_frame.track_id > 0:
        kp_color   = TRACK_COLORS[(pose_frame.track_id - 1) % len(TRACK_COLORS)]
        bone_color = kp_color

    kp  = pose_frame.keypoints
    pts = (kp[:, :2] * np.array([w, h])).astype(int)
    vis = kp[:, 2] > 0.2

    for a, b in SKELETON_EDGES:
        if vis[a] and vis[b]:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), bone_color, 2)
    for i in range(NUM_JOINTS):
        if vis[i]:
            cv2.circle(out, tuple(pts[i]), 4, kp_color, -1)

    # Draw track ID label near head
    if pose_frame.track_id > 0 and vis[0]:
        cv2.putText(out, f"#{pose_frame.track_id}",
                    (pts[0][0] - 10, pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, kp_color, 2)
    return out


def draw_all_poses(bgr_frame: np.ndarray,
                   poses: list[PoseFrame]) -> np.ndarray:
    out = bgr_frame.copy()
    for p in poses:
        out = draw_pose(out, p)
    return out
