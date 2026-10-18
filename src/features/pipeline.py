"""Sequence -> sliding-window feature table pipeline.

Per sequence: read frames -> detect persons -> track (ByteTrack) -> optical
flow -> per-frame signals -> sliding windows -> schema-ordered feature rows.

Honesty notes:
- Window label = 1 if ANY frame in the window is anomalous (UCSD frame GT).
- Partial trailing windows are dropped (full windows only).
- Speeds are px/frame in the processed resolution (UCSD frames are tiny;
  uploads may be resized by ``max_dim``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.datasets.ucsd_anomaly import AnomalySequence
from src.detection.detector import PersonDetector
from src.features.extract import FrameSignals, extract_window_features, optical_flow_stats
from src.features.schema import FeatureSchema
from src.preprocessing.video import read_frame
from src.tracking.tracker import CrowdTracker, TrackedPerson
from src.tracking.trajectories import TrajectoryStore

META_COLUMNS = ["sequence_id", "part", "split", "window_start", "window_end", "label"]


@dataclass(frozen=True)
class WindowConfig:
    window: int = 30
    stride: int = 5
    grid_size: int = 4
    max_dim: int | None = 1280  # resize frames longer side (None = keep)


def _centers_to_grid(centers: np.ndarray, shape_hw: tuple[int, int], grid: int) -> np.ndarray:
    counts = np.zeros((grid, grid), dtype=np.int64)
    if centers.size == 0:
        return counts
    h, w = shape_hw
    cols = np.clip((centers[:, 0] / w * grid).astype(int), 0, grid - 1)
    rows = np.clip((centers[:, 1] / h * grid).astype(int), 0, grid - 1)
    np.add.at(counts, (rows, cols), 1)
    return counts


def _maybe_resize(frame: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))


def process_sequence(
    seq: AnomalySequence,
    detector: PersonDetector | None = None,
    config: WindowConfig = WindowConfig(),
    max_frames: int | None = None,
) -> list[dict]:
    """Run the full pipeline on one sequence; returns schema-ordered rows."""
    if seq.n_frames == 0:
        return []
    if detector is None:
        detector = PersonDetector()
    schema = FeatureSchema.default()

    n = seq.n_frames if max_frames is None else min(seq.n_frames, max_frames)
    frame_files = seq.frame_files[:n]
    labels = seq.frame_labels[:n]

    tracker: CrowdTracker | None = None
    store_all = TrajectoryStore()
    persons_by_frame: dict[int, list[TrackedPerson]] = {}
    signals: list[FrameSignals] = []
    prev_gray: np.ndarray | None = None

    for idx, fpath in enumerate(frame_files):
        frame = _maybe_resize(read_frame(fpath), config.max_dim)
        h, w = frame.shape[:2]
        if tracker is None:
            tracker = CrowdTracker(frame_width=w, frame_height=h)
        result = detector.detect(frame)
        persons = tracker.update(result.detections)
        persons_by_frame[idx] = persons
        store_all.add_frame(idx, persons)

        centers = np.array([p.center for p in persons], dtype=float).reshape(-1, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            flow_mean, flow_std = 0.0, 0.0
        else:
            flow_mean, flow_std = optical_flow_stats(prev_gray, gray)
        prev_gray = gray
        signals.append(
            FrameSignals(
                frame_index=idx,
                count=len(result.detections),
                grid_counts=_centers_to_grid(centers, (h, w), config.grid_size),
                centers=centers,
                flow_mean=flow_mean,
                flow_std=flow_std,
            )
        )

    rows: list[dict] = []
    starts = range(0, n - config.window + 1, config.stride)
    for start in starts:
        end = start + config.window  # exclusive
        window_store = TrajectoryStore()
        for fi in range(start, end):
            window_store.add_frame(fi, persons_by_frame.get(fi, []))
        feats = extract_window_features(
            signals[start:end], window_store, schema=schema, frame_shape=(h, w)
        )
        row = {
            "sequence_id": seq.sequence_id,
            "part": seq.part,
            "split": seq.split,
            "window_start": start,
            "window_end": end,
            "label": int(labels[start:end].any()),
        }
        row.update(feats)
        rows.append(row)
    return rows


def build_feature_table(
    adapter,
    detector: PersonDetector | None = None,
    config: WindowConfig = WindowConfig(),
    split: str | None = None,
    part: str | None = None,
    max_sequences: int | None = None,
    max_frames_per_sequence: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Process all (or filtered) sequences into one DataFrame."""
    rows: list[dict] = []
    for i, seq in enumerate(adapter.iter_sequences(split=split, part=part)):
        if max_sequences is not None and i >= max_sequences:
            break
        rows.extend(
            process_sequence(seq, detector=detector, config=config, max_frames=max_frames_per_sequence)
        )
        if progress:
            print(f"[prepare] {seq.sequence_id}: {seq.n_frames} frames -> cumulative {len(rows)} windows")
    if not rows:
        return pd.DataFrame(columns=META_COLUMNS + FeatureSchema.default().names)
    df = pd.DataFrame(rows)
    return df[META_COLUMNS + FeatureSchema.default().names]
