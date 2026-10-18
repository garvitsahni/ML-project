"""Canonical feature schema: names, order, units, limitations.

``FeatureSchema.names`` is the SINGLE ordering source consumed by training,
saved bundles, and inference — schema mismatches are rejected at load time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str  # "float"
    unit: str
    limitation: str
    group: str  # "static" | "movement"


_STATIC = [
    ("person_count", "count (detector estimate)", "YOLO person boxes, no counting GT available"),
    ("density_per_mpx", "persons / megapixel", "image-area density, no scene area calibration"),
    ("grid_concentration", "fraction 0-1", "max grid cell share; depends on grid_size"),
    ("spread_x", "normalized 0-~1", "std of x centers / image width, pixels only"),
    ("spread_y", "normalized 0-~1", "std of y centers / image height, pixels only"),
    ("grid_entropy", "normalized 0-1", "grid cell occupancy entropy; depends on grid_size"),
]

_MOVEMENT = [
    ("density_delta", "persons/frame", "signed count change across the window"),
    ("mean_speed_pxf", "px/frame", "track center speed; NEVER m/s (no scale)"),
    ("speed_std_pxf", "px/frame", "speed variability; 0 with <2 samples"),
    ("direction_entropy", "normalized 0-1", "movement direction entropy (8 bins)"),
    ("directional_consistency", "0-1", "1 = all one direction; 0 = canceling flows"),
    ("flow_mean_magnitude", "px/frame", "Farneback optical-flow magnitude"),
    ("flow_std_pxf", "px/frame", "within-window optical-flow variability"),
    ("track_count", "count", "unique ByteTrack ids in window"),
    ("track_persistence", "fraction 0-1", "mean track visibility across window frames"),
]


class FeatureSchema:
    """Frozen canonical schema; ``default()`` returns the project instance."""

    def __init__(self) -> None:
        specs = [FeatureSpec(n, "float", u, lim, "static") for n, u, lim in _STATIC]
        specs += [FeatureSpec(n, "float", u, lim, "movement") for n, u, lim in _MOVEMENT]
        self.specs: tuple[FeatureSpec, ...] = tuple(specs)
        self.names: list[str] = [s.name for s in self.specs]

    @classmethod
    def default(cls) -> "FeatureSchema":
        return _DEFAULT

    def names_for(self, groups: set[str]) -> list[str]:
        return [s.name for s in self.specs if s.group in groups]

    @property
    def static_names(self) -> list[str]:
        return self.names_for({"static"})

    @property
    def movement_names(self) -> list[str]:
        return self.names_for({"movement"})


_DEFAULT = FeatureSchema()
