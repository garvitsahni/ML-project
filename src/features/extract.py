"""Window-level feature extraction: per-frame signals -> schema-ordered dict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from src.features.movement import (
    directional_consistency,
    direction_entropy,
    movement_speeds_from_trajectories,
)
from src.features.schema import FeatureSchema
from src.features.static import static_features
from src.tracking.trajectories import TrajectoryStore


@dataclass(frozen=True)
class FrameSignals:
    """Per-frame inputs to window aggregation (no images retained)."""

    frame_index: int
    count: int
    grid_counts: np.ndarray  # (grid, grid)
    centers: np.ndarray  # (n, 2) xy pixels
    flow_mean: float  # Farneback mean magnitude vs previous frame (0 for first)
    flow_std: float
    flow_dir_inconsistency: float = 0.0  # 1 - |mean unit flow dir| (0 for first)


def optical_flow_stats(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float, float]:
    """Farneback dense flow -> (mean magnitude, std magnitude, direction inconsistency).

    Direction inconsistency = 1 - |mean unit direction| over pixels moving
    > 0.5 px/frame; 0.0 when no qualifying pixels.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.linalg.norm(flow, axis=2)
    moving = mag > 0.5
    if not moving.any():
        return float(mag.mean()), float(mag.std()), 0.0
    units = flow[moving] / mag[moving, None]
    dir_inc = 1.0 - float(np.linalg.norm(units.mean(axis=0)))
    return float(mag.mean()), float(mag.std()), dir_inc


def _third_mean(values: np.ndarray) -> float:
    k = max(1, len(values) // 3)
    return float(values[-k:].mean()) - float(values[:k].mean())


def extract_window_features(
    signals: Sequence[FrameSignals],
    store: TrajectoryStore,
    schema: FeatureSchema,
    frame_shape: tuple[int, int],
) -> dict[str, float]:
    """Aggregate a window of per-frame signals into one schema-ordered row.

    ``store`` must contain ONLY the window's track observations (window-local),
    so persistence and speeds are scoped to the window. All-zero on empty input.
    """
    feats: dict[str, float] = {name: 0.0 for name in schema.names}

    if not signals:
        return feats

    counts = np.array([s.count for s in signals], dtype=float)
    h, w = frame_shape

    # ---- static group
    per_frame_static = [
        static_features(s.count, s.grid_counts, s.centers, frame_shape, s.grid_counts.shape[0])
        for s in signals
    ]
    feats["person_count"] = float(counts.mean())
    feats["density_per_mpx"] = float(counts.mean()) / (h * w / 1e6)
    feats["grid_concentration"] = float(np.mean([f["grid_concentration"] for f in per_frame_static]))
    feats["grid_entropy"] = float(np.mean([f["grid_entropy"] for f in per_frame_static]))
    pooled = np.vstack([s.centers for s in signals if s.centers.size]) if any(
        s.centers.size for s in signals
    ) else np.zeros((0, 2))
    if len(pooled) >= 2:
        feats["spread_x"] = float(np.std(pooled[:, 0]) / w)
        feats["spread_y"] = float(np.std(pooled[:, 1]) / h)

    # ---- movement group
    feats["density_delta"] = _third_mean(counts)
    track_ids = store.track_ids()
    speeds = movement_speeds_from_trajectories(store, track_ids)
    if speeds.size:
        feats["mean_speed_pxf"] = float(speeds.mean())
        feats["speed_std_pxf"] = float(speeds.std()) if speeds.size >= 2 else 0.0
    feats["direction_entropy"] = direction_entropy(store, track_ids)
    feats["directional_consistency"] = directional_consistency(store, track_ids)
    feats["flow_mean_magnitude"] = float(np.mean([s.flow_mean for s in signals]))
    feats["flow_std_pxf"] = float(np.mean([s.flow_std for s in signals]))
    feats["flow_direction_inconsistency"] = float(
        np.mean([s.flow_dir_inconsistency for s in signals])
    )
    feats["track_count"] = float(len(track_ids))
    if track_ids:
        feats["track_persistence"] = float(
            np.mean([store.persistence(tid) for tid in track_ids])
        )

    return feats
