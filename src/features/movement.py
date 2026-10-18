"""Movement features derived from trajectories (px/frame units only)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.tracking.trajectories import TrajectoryStore

N_DIRECTION_BINS = 8


def _unit_steps(store: TrajectoryStore, track_ids: Sequence[int]) -> np.ndarray:
    """(m, 2) unit direction vectors for consecutive-frame steps."""
    vectors = []
    for tid in track_ids:
        traj = store.trajectory(tid)
        if len(traj) < 2:
            continue
        dxy = np.diff(traj[:, 1:], axis=0)
        dframe = np.diff(traj[:, 0])
        mask = dframe == 1
        dxy = dxy[mask]
        dxy = dxy[np.linalg.norm(dxy, axis=1) > 1e-9]
        if dxy.size:
            norms = np.linalg.norm(dxy, axis=1, keepdims=True)
            vectors.append(dxy / norms)
    if not vectors:
        return np.zeros((0, 2))
    return np.vstack(vectors)


def movement_speeds_from_trajectories(
    store: TrajectoryStore, track_ids: Sequence[int]
) -> np.ndarray:
    """All consecutive-frame speeds (px/frame) across the given tracks."""
    speeds = [store.speeds(tid) for tid in track_ids]
    speeds = [s for s in speeds if s.size]
    return np.concatenate(speeds) if speeds else np.zeros(0)


def directional_consistency(store: TrajectoryStore, track_ids: Sequence[int]) -> float:
    """Norm of the mean unit direction vector: 1 = unidirectional, 0 = canceling."""
    units = _unit_steps(store, track_ids)
    if units.size == 0:
        return 0.0
    return float(np.linalg.norm(units.mean(axis=0)))


def direction_entropy(store: TrajectoryStore, track_ids: Sequence[int]) -> float:
    """Normalized entropy of movement directions over 8 angular bins (0-1)."""
    units = _unit_steps(store, track_ids)
    if units.size == 0:
        return 0.0
    angles = np.arctan2(units[:, 1], units[:, 0])  # [-pi, pi]
    hist, _ = np.histogram(angles, bins=N_DIRECTION_BINS, range=(-math.pi, math.pi))
    probs = hist.astype(float) / hist.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    return entropy / math.log(N_DIRECTION_BINS)
