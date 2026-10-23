"""Shared incremental crowd-analysis loop (detect -> track -> flow -> features).

The Video Analysis and Live Camera pages previously duplicated this loop
inline; ``FrameAnalyzer`` is the extracted state machine. Frames are fed one
at a time in any chunking (the fragment-based pages feed them a batch per
rerun); tracker/signal state persists across calls until :meth:`reset`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class FrameAnalyzer:
    """Incremental analyzer producing per-frame signals and window features."""

    def __init__(
        self,
        *,
        grid_size: int = 4,
        max_dim: int | None = 1280,
        window: int = 30,
        window_stride: int = 5,
        detector: Any = None,
        tracker_factory: Any = None,
        schema: Any = None,
    ) -> None:
        import cv2

        from src.features.pipeline import WindowConfig

        self._cv2 = cv2
        self._cfg = WindowConfig(
            window=window, stride=window_stride,
            grid_size=grid_size, max_dim=max_dim,
        )
        self._detector = detector
        self._tracker_factory = tracker_factory
        self._schema = schema
        self.reset()

    def reset(self) -> None:
        """Clear all stream state (tracker, signals, flow history)."""
        self._tracker: Any = None
        self._signals: list = []
        self._persons_by_frame: dict[int, list] = {}
        self._prev_gray: np.ndarray | None = None
        self._frame_shape: tuple[int, int] = (0, 0)
        self._processed = 0

    @property
    def processed_frames(self) -> int:
        """Number of frames consumed since construction/reset."""
        return self._processed

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(h, w) of the most recently processed (resized) frame."""
        return self._frame_shape

    def add_frame(self, frame: np.ndarray) -> dict:
        """Process one BGR frame.

        Returns a dict with ``frame_index``, ``count`` (detector boxes),
        ``persons`` (tracker output), ``flow_mean``/``flow_std`` (0.0 for the
        first frame), ``shape`` (resized h, w) and ``window`` — the feature
        row of a completed stride-window, or None when no window completed
        on this frame.
        """
        from src.detection.detector import PersonDetector
        from src.features.extract import FrameSignals, optical_flow_stats
        from src.features.pipeline import _centers_to_grid, _maybe_resize
        from src.tracking.tracker import CrowdTracker

        cfg = self._cfg
        frame = _maybe_resize(frame, cfg.max_dim)
        h, w = frame.shape[:2]
        self._frame_shape = (h, w)

        if self._tracker is None:
            factory = self._tracker_factory or CrowdTracker
            self._tracker = factory(frame_width=w, frame_height=h)
        if self._detector is None:
            self._detector = PersonDetector()

        result = self._detector.detect(frame)
        persons = self._tracker.update(result.detections)

        idx = len(self._signals)
        self._persons_by_frame[idx] = persons
        centers = np.array([p.center for p in persons],
                           dtype=np.float64).reshape(-1, 2)
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            flow_mean, flow_std = 0.0, 0.0
        else:
            flow_mean, flow_std = optical_flow_stats(self._prev_gray, gray)
        self._prev_gray = gray

        self._signals.append(FrameSignals(
            frame_index=idx,
            count=len(result.detections),
            grid_counts=_centers_to_grid(centers, (h, w), cfg.grid_size),
            centers=centers,
            flow_mean=flow_mean,
            flow_std=flow_std,
        ))
        self._processed = idx + 1

        window_row = None
        total = len(self._signals)
        if total >= cfg.window and (total - cfg.window) % cfg.stride == 0:
            window_row = self._window_row(total - cfg.window)

        return {
            "frame_index": idx,
            "count": len(result.detections),
            "persons": persons,
            "flow_mean": flow_mean,
            "flow_std": flow_std,
            "shape": (h, w),
            "window": window_row,
        }

    def _window_row(self, start: int) -> dict:
        """Feature row for the window [start, start + window)."""
        from src.features.extract import extract_window_features
        from src.features.schema import FeatureSchema
        from src.tracking.trajectories import TrajectoryStore

        cfg = self._cfg
        end = start + cfg.window
        store = TrajectoryStore()
        for fi in range(cfg.window):
            store.add_frame(fi, self._persons_by_frame.get(start + fi, []))
        schema = self._schema or FeatureSchema.default()
        feats = extract_window_features(
            self._signals[start:end], store, schema=schema,
            frame_shape=self._frame_shape,
        )
        return {"window_start": start, "window_end": end, **feats}

    def latest_window_features(self) -> dict | None:
        """Rolling feature row of the most recent full window, else None."""
        total = len(self._signals)
        if total < self._cfg.window:
            return None
        return self._window_row(total - self._cfg.window)


def draw_persons(frame: np.ndarray, persons: Sequence[Any],
                 *, color: tuple[int, int, int] = (191, 212, 45)) -> np.ndarray:
    """Copy of ``frame`` with tracked-person boxes and track IDs (BGR)."""
    import cv2

    out = frame.copy()
    for person in persons:
        p1 = (int(person.x1), int(person.y1))
        p2 = (int(person.x2), int(person.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cx, cy = person.center
        cv2.putText(out, f"#{person.track_id}", (int(cx) - 10, int(cy) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
