"""Detector calibration: pick conf/imgsz by pre-declared recall rule.

Selection rule (spec Task 4): among configs whose projected full-rebuild
time fits the budget, pick the highest mean detections per frame; ties
break toward the faster config. Detections are detector estimates —
there is no counting ground truth in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

TOTAL_FRAMES = 18560  # full UCSD v1p2 (98 sequences), for time projection


def project_rebuild_hours(mean_seconds_per_frame: float, n_frames: int = TOTAL_FRAMES) -> float:
    """Projected wall-clock hours for a full feature-table rebuild."""
    return float(mean_seconds_per_frame) * n_frames / 3600.0


def sample_frame_files(adapter, n_frames: int = 40, seed: int = 42) -> list[Path]:
    """Deterministically sample ``n_frames`` frames from the TRAIN split only.

    Even stride over the concatenated frame list; ``seed`` kept for API
    stability (sampling is fully deterministic without it).
    """
    all_files: list[Path] = [
        f for seq in adapter.iter_sequences(split="train") for f in seq.frame_files
    ]
    if not all_files:
        raise ValueError("No train-split frames available to sample")
    if n_frames >= len(all_files):
        return sorted(all_files)
    step = len(all_files) / n_frames
    picked = {all_files[int(i * step)] for i in range(n_frames)}
    return sorted(picked)


def select_config(results: Iterable[dict], max_rebuild_hours: float = 2.0) -> dict:
    """Pre-declared automatic selection: max recall within the rebuild budget.

    Each result: {"conf", "imgsz", "mean_detections_per_frame",
    "mean_seconds_per_frame", "projected_rebuild_hours", "overlay_files"}.
    """
    results = list(results)
    if not results:
        raise ValueError("No calibration results to select from")
    within = [r for r in results if r["projected_rebuild_hours"] <= max_rebuild_hours]
    if not within:
        raise ValueError(
            f"No config fits the {max_rebuild_hours}h rebuild budget "
            f"(fastest projects {min(r['projected_rebuild_hours'] for r in results):.1f}h)"
        )
    within.sort(
        key=lambda r: (-r["mean_detections_per_frame"], r["mean_seconds_per_frame"])
    )
    return within[0]
