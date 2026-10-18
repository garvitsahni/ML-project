"""ByteTrack wrapper (ultralytics) over per-frame person detections.

Tracks operate in PIXELS; speeds derived from trajectories are px/frame —
never converted to m/s (no real-world scale is available).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.detection.detector import Detection


class TrackerUnavailableError(RuntimeError):
    """Raised when the ultralytics tracker backend cannot be loaded."""


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def _bytetrack_args():
    from types import SimpleNamespace

    import ultralytics
    import yaml

    cfg = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    if not cfg.is_file():
        raise TrackerUnavailableError(f"ByteTrack config not found: {cfg}")
    with cfg.open("r", encoding="utf-8") as fh:
        return SimpleNamespace(**yaml.safe_load(fh))


class CrowdTracker:
    """Stateful ByteTrack tracker; call ``update`` once per frame in order."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        frame_rate: int = 30,
        track_buffer: int | None = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_rate = frame_rate
        self._args = _bytetrack_args()
        if track_buffer is not None:
            self._args.track_buffer = track_buffer
        self._tracker = self._new_tracker()
        self.frame_count = 0

    def _new_tracker(self):
        from ultralytics.trackers import BYTETracker

        return BYTETracker(self._args, frame_rate=self.frame_rate)

    def reset(self) -> None:
        self._tracker = self._new_tracker()
        self.frame_count = 0

    def update(self, detections: Sequence[Detection]) -> list[TrackedPerson]:
        """Advance one frame; returns currently active tracks."""
        import torch
        from ultralytics.engine.results import Boxes

        self.frame_count += 1
        if detections:
            rows = [[d.x1, d.y1, d.x2, d.y2, d.conf, float(d.cls_id)] for d in detections]
            data = torch.tensor(rows, dtype=torch.float32)
        else:
            data = torch.zeros((0, 6), dtype=torch.float32)
        boxes = Boxes(data, (self.frame_height, self.frame_width))
        # BYTETracker.update expects a Boxes-like with .conf/.xywh/.cls/.xyxy
        out = self._tracker.update(boxes)
        people: list[TrackedPerson] = []
        if out is None or len(out) == 0:
            return people
        # columns: x1, y1, x2, y2, track_id, conf, cls, idx
        for row in np.asarray(out):
            people.append(
                TrackedPerson(
                    track_id=int(row[4]),
                    x1=float(row[0]),
                    y1=float(row[1]),
                    x2=float(row[2]),
                    y2=float(row[3]),
                    conf=float(row[5]),
                )
            )
        return people
