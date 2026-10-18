"""Trajectory storage and per-track kinematics (px/frame units only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.tracking.tracker import TrackedPerson


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    track_id: int
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


class TrajectoryStore:
    """Collects per-frame track observations; computes speeds and persistence.

    Units: positions in pixels, speeds in px/frame (consecutive frames only —
    gaps produce no speed samples, avoiding phantom velocities).
    """

    def __init__(self) -> None:
        self._points: dict[int, list[TrackPoint]] = {}
        self._frames: set[int] = set()

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    def add_frame(self, frame_index: int, persons: Sequence[TrackedPerson]) -> None:
        self._frames.add(frame_index)
        for p in persons:
            cx, cy = p.center
            self._points.setdefault(p.track_id, []).append(
                TrackPoint(
                    frame_index=frame_index,
                    track_id=p.track_id,
                    cx=cx,
                    cy=cy,
                    x1=p.x1,
                    y1=p.y1,
                    x2=p.x2,
                    y2=p.y2,
                    conf=p.conf,
                )
            )

    def track_ids(self) -> list[int]:
        return sorted(self._points.keys())

    def trajectory(self, track_id: int) -> np.ndarray:
        """(n, 3) array of [frame_index, cx, cy] sorted by frame."""
        pts = sorted(self._points.get(track_id, []), key=lambda p: p.frame_index)
        return np.array([[p.frame_index, p.cx, p.cy] for p in pts], dtype=np.float64)

    def speeds(self, track_id: int) -> np.ndarray:
        """Per-step speeds in px/frame between CONSECUTIVE frames only."""
        traj = self.trajectory(track_id)
        if len(traj) < 2:
            return np.zeros(0, dtype=np.float64)
        frames = traj[:, 0]
        dxy = np.diff(traj[:, 1:], axis=0)
        dframe = np.diff(frames)
        mask = dframe == 1  # consecutive frames only
        if not mask.any():
            return np.zeros(0, dtype=np.float64)
        return np.linalg.norm(dxy[mask], axis=1) / dframe[mask]

    def mean_speed(self, track_id: int) -> float:
        speeds = self.speeds(track_id)
        return float(speeds.mean()) if speeds.size else float("nan")

    def persistence(self, track_id: int) -> float:
        """Fraction of observed frames in which this track appears."""
        if self.total_frames == 0:
            return 0.0
        return len(self._points.get(track_id, [])) / self.total_frames
