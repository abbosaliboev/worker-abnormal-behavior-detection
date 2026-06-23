"""
UP-Fall Dataset loader.

Expected directory layout (set UPFALL_DATASET_PATH in config.py):

  UP-Fall/
  ├── Subject01/
  │   ├── Activity01/          # falling_forward_hands
  │   │   ├── Trial01/
  │   │   │   └── cam1.mp4    (or .avi)
  │   │   └── Trial02/
  │   │       └── cam1.mp4
  │   ├── Activity02/
  │   │   ...
  │   └── Activity11/
  ├── Subject02/
  │   ...
  └── Subject17/

The loader discovers all video files and returns a flat list of VideoClip objects
with subject, activity, and trial metadata attached.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    UPFALL_DATASET_PATH,
    UPFALL_ACTIVITIES,
    FALL_ACTIVITY_IDS,
    RUNNING_ACTIVITY_IDS,
    INACTIVITY_ACTIVITY_IDS,
    EVAL,
)


@dataclass
class VideoClip:
    path: str
    subject_id: int
    activity_id: int
    trial_id: int
    activity_name: str
    is_fall: bool
    is_running: bool
    is_inactivity: bool


def _find_video(trial_dir: str) -> Optional[str]:
    """Return first video file found in a trial directory."""
    for ext in ("*.mp4", "*.avi", "*.MP4", "*.AVI", "*.mkv"):
        matches = list(Path(trial_dir).glob(ext))
        if matches:
            return str(matches[0])
    # Some releases put videos directly under trial dir without extension filter
    for f in Path(trial_dir).iterdir():
        if f.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            return str(f)
    return None


def load_upfall_clips(dataset_path: Optional[str] = None) -> list[VideoClip]:
    """
    Scan the UP-Fall directory tree and return all VideoClip objects.

    Handles both naming conventions seen in different releases:
      Subject01/Activity01/Trial01/cam1.mp4
      S01/A01/T01/video.mp4
    """
    root = Path(dataset_path or UPFALL_DATASET_PATH)
    if not root.exists():
        raise FileNotFoundError(
            f"UP-Fall dataset not found at '{root}'. "
            "Set UPFALL_DATASET_PATH in src/config.py."
        )

    clips: list[VideoClip] = []

    # Match subject directories (Subject01 / S01 / subject_01 etc.)
    subj_re = re.compile(r"(?:subject|s)[-_]?(\d+)", re.IGNORECASE)
    act_re  = re.compile(r"(?:activity|act|a)[-_]?(\d+)", re.IGNORECASE)
    trial_re= re.compile(r"(?:trial|t)[-_]?(\d+)", re.IGNORECASE)

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sm = subj_re.search(subj_dir.name)
        if sm is None:
            continue
        subj_id = int(sm.group(1))

        for act_dir in sorted(subj_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            am = act_re.search(act_dir.name)
            if am is None:
                continue
            act_id = int(am.group(1))
            if act_id not in UPFALL_ACTIVITIES:
                continue

            for trial_dir in sorted(act_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                tm = trial_re.search(trial_dir.name)
                trial_id = int(tm.group(1)) if tm else 1

                video_path = _find_video(str(trial_dir))
                if video_path is None:
                    continue

                clips.append(VideoClip(
                    path=video_path,
                    subject_id=subj_id,
                    activity_id=act_id,
                    trial_id=trial_id,
                    activity_name=UPFALL_ACTIVITIES[act_id],
                    is_fall=act_id in FALL_ACTIVITY_IDS,
                    is_running=act_id in RUNNING_ACTIVITY_IDS,
                    is_inactivity=act_id in INACTIVITY_ACTIVITY_IDS,
                ))

    if not clips:
        raise RuntimeError(
            f"No video clips found under '{root}'. "
            "Check your directory structure — expected Subject*/Activity*/Trial*/video.mp4"
        )

    print(f"Loaded {len(clips)} clips from UP-Fall ({len({c.subject_id for c in clips})} subjects)")
    return clips


def train_test_split_by_subject(clips: list[VideoClip]
                                ) -> tuple[list[VideoClip], list[VideoClip]]:
    """
    Split clips by subject ID (group-aware) so no subject appears in both
    train and test sets. This avoids data leakage when calibrating thresholds.
    """
    subjects = np.array([c.subject_id for c in clips])
    indices  = np.arange(len(clips))

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=EVAL["test_subjects_fraction"],
        random_state=EVAL["random_seed"],
    )
    train_idx, test_idx = next(gss.split(indices, groups=subjects))

    train = [clips[i] for i in train_idx]
    test  = [clips[i] for i in test_idx]

    train_subj = {c.subject_id for c in train}
    test_subj  = {c.subject_id for c in test}
    print(f"Train: {len(train)} clips, subjects {sorted(train_subj)}")
    print(f"Test:  {len(test)}  clips, subjects {sorted(test_subj)}")
    return train, test


def filter_clips(clips: list[VideoClip],
                 task: str) -> tuple[list[VideoClip], list[VideoClip]]:
    """
    Return (positive_clips, negative_clips) for a given task.

    task: "fall" | "running" | "inactivity"
    """
    if task == "fall":
        pos = [c for c in clips if c.is_fall]
        neg = [c for c in clips if not c.is_fall]
    elif task == "running":
        pos = [c for c in clips if c.is_running]
        neg = [c for c in clips if not c.is_running]
    elif task == "inactivity":
        pos = [c for c in clips if c.is_inactivity]
        neg = [c for c in clips if not c.is_inactivity]
    else:
        raise ValueError(f"Unknown task '{task}'. Use 'fall', 'running', or 'inactivity'.")

    return pos, neg


def summarize(clips: list[VideoClip]):
    """Print a quick count table by activity."""
    from collections import Counter
    counts = Counter(c.activity_name for c in clips)
    print("\nActivity distribution:")
    for name, n in sorted(counts.items()):
        print(f"  {name:<35s} {n:3d}")
    print(f"  {'TOTAL':<35s} {len(clips):3d}\n")
