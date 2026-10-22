"""Movement features derived from trajectories (px/frame units only)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.tracking.trajectories import TrajectoryStore

N_DIRECTION_BINS = 8
FAST_MOVER_K = 3.0
STOP_SPEED_FACTOR = 2.0
STOP_REL_SPEED = 0.25
REVERSAL_DOT = -0.36
MIN_SPEED_SAMPLES_FOR_STOP = 3


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


def nearest_neighbor_stats(
    store: TrajectoryStore,
    track_ids: Sequence[int],
    centers_by_frame: dict[int, np.ndarray],
    image_diagonal: float,
) -> tuple[float, float]:
    """(mean, std) of per-frame mean nearest-neighbor distance, normalized by
    ``image_diagonal``. Frames with <2 centers are skipped; all-degenerate -> (0, 0)."""
    per_frame_means = []
    for centers in centers_by_frame.values():
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        if len(centers) < 2:
            continue
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        per_frame_means.append(float(dists.min(axis=1).mean()))
    if not per_frame_means:
        return 0.0, 0.0
    values = np.asarray(per_frame_means, dtype=float)
    if image_diagonal <= 0:
        return 0.0, 0.0
    return float(values.mean() / image_diagonal), float(values.std() / image_diagonal)


def abrupt_stop_fraction(
    store: TrajectoryStore, track_ids: Sequence[int], median_window_speed: float
) -> float:
    """Fraction of tracks (with >= MIN_SPEED_SAMPLES_FOR_STOP samples) whose
    max speed is scene-fast and whose final speed dropped to <= STOP_REL_SPEED
    of that max. 0.0 when no track qualifies for evaluation."""
    floor = max(float(median_window_speed), 1e-6)
    eligible = 0
    stops = 0
    for tid in track_ids:
        speeds = store.speeds(tid)
        if speeds.size < MIN_SPEED_SAMPLES_FOR_STOP:
            continue
        eligible += 1
        smax = float(speeds.max())
        if smax >= STOP_SPEED_FACTOR * floor and speeds[-1] <= STOP_REL_SPEED * smax:
            stops += 1
    if eligible == 0:
        return 0.0
    return stops / eligible


def direction_reversal_rate(store: TrajectoryStore, track_ids: Sequence[int]) -> float:
    """Reversals (consecutive unit steps with dot < REVERSAL_DOT) per step pair."""
    reversals = 0
    pairs = 0
    for tid in track_ids:
        traj = store.trajectory(tid)
        if len(traj) < 3:
            continue
        dxy = np.diff(traj[:, 1:], axis=0)
        dframe = np.diff(traj[:, 0])
        mask = dframe == 1
        dxy = dxy[mask]
        norms = np.linalg.norm(dxy, axis=1)
        dxy = dxy[norms > 1e-9]
        if len(dxy) < 2:
            continue
        units = dxy / np.linalg.norm(dxy, axis=1, keepdims=True)
        dots = (units[:-1] * units[1:]).sum(axis=1)
        reversals += int((dots < REVERSAL_DOT).sum())
        pairs += int(dots.size)
    if pairs == 0:
        return 0.0
    return reversals / pairs


def fast_mover_fraction(
    store: TrajectoryStore, track_ids: Sequence[int], median_window_speed: float
) -> float:
    """Fraction of tracks (with >=1 speed sample) whose mean speed exceeds
    FAST_MOVER_K x the scene median speed."""
    floor = max(float(median_window_speed), 1e-6)
    eligible = 0
    fast = 0
    for tid in track_ids:
        speeds = store.speeds(tid)
        if speeds.size == 0:
            continue
        eligible += 1
        if float(speeds.mean()) > FAST_MOVER_K * floor:
            fast += 1
    if eligible == 0:
        return 0.0
    return fast / eligible


def speed_p90(store: TrajectoryStore, track_ids: Sequence[int]) -> float:
    """90th percentile of pooled consecutive-frame speeds (px/frame)."""
    speeds = movement_speeds_from_trajectories(store, track_ids)
    if speeds.size == 0:
        return 0.0
    return float(np.percentile(speeds, 90))


def track_turnover_rate(
    store: TrajectoryStore, track_ids: Sequence[int], window_start: int, window_end: int
) -> float:
    """(births + deaths) per frame in the window. Birth = first observation after
    ``window_start``; death = last observation before ``window_end - 1``."""
    length = window_end - window_start
    if length <= 0 or not track_ids:
        return 0.0
    events = 0
    for tid in track_ids:
        traj = store.trajectory(tid)
        if len(traj) == 0:
            continue
        first_frame, last_frame = traj[0, 0], traj[-1, 0]
        if first_frame > window_start:
            events += 1
        if last_frame < window_end - 1:
            events += 1
    return events / length
