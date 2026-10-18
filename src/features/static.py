"""Static (density/layout) features for one frame."""

from __future__ import annotations

import math

import numpy as np


def static_features(
    count: int,
    grid_counts: np.ndarray,
    centers: np.ndarray,
    frame_shape: tuple[int, int],
    grid_size: int,
) -> dict[str, float]:
    """Compute the static feature group for a single frame.

    ``grid_counts``: (grid, grid) int matrix of box-center occupancy.
    ``centers``: (n, 2) array of (cx, cy) in pixels.
    """
    h, w = frame_shape
    feats: dict[str, float] = {
        "person_count": float(count),
        "density_per_mpx": float(count) / (h * w / 1e6),
    }

    if count > 0:
        feats["grid_concentration"] = float(grid_counts.max()) / float(count)
    else:
        feats["grid_concentration"] = 0.0

    if count >= 2 and centers.size:
        feats["spread_x"] = float(np.std(centers[:, 0]) / w)
        feats["spread_y"] = float(np.std(centers[:, 1]) / h)
    else:
        feats["spread_x"] = 0.0
        feats["spread_y"] = 0.0

    total_cells = grid_size * grid_size
    if count > 0:
        probs = grid_counts.ravel().astype(float) / float(count)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
        feats["grid_entropy"] = entropy / math.log(total_cells) if total_cells > 1 else 0.0
    else:
        feats["grid_entropy"] = 0.0

    return feats
