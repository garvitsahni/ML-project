"""Canonical feature schema: names, order, units, limitations.

``FeatureSchema.names`` is the SINGLE ordering source consumed by training,
saved bundles, and inference — schema mismatches are rejected at load time.
``FeatureSchema.fingerprint`` content-hashes that ordering; bundles record
it so inference can distinguish "trained under this schema" (possibly on a
screened subset) from "predates this schema".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def schema_fingerprint(names: list[str]) -> str:
    """Stable sha256 over an ordered feature-name list (schema identity)."""
    return hashlib.sha256("\x1f".join(names).encode("utf-8")).hexdigest()


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
    ("nn_distance_mean", "normalized 0-1", "mean nearest-neighbor distance / image diagonal; 0 with <2 centers"),
    ("nn_distance_std", "normalized 0-1", "frame-to-frame variability of mean NN distance"),
    ("abrupt_stop_fraction", "fraction 0-1", "tracks decelerating hard within window; 0 with <3 speed samples"),
    ("direction_reversal_rate", "fraction 0-1", "consecutive-step direction flips >110 deg per step pair"),
    ("fast_mover_fraction", "fraction 0-1", "tracks >3x scene median speed (intruder proxy)"),
    ("speed_p90", "px/frame", "90th percentile track speed; NEVER m/s"),
    ("track_turnover_rate", "tracks/frame", "track births+deaths per frame in window"),
    ("flow_direction_inconsistency", "normalized 0-1", "1 - |mean flow direction| over motion pixels; 0 with no motion"),
]


# The pre-schema-v2 feature set (15 names; git history of this file). Single
# source for baseline reproduction (attribution config 1) and the redundancy
# screen's comparison set.
ORIGINAL_15 = [
    "person_count", "density_per_mpx", "grid_concentration", "spread_x",
    "spread_y", "grid_entropy", "density_delta", "mean_speed_pxf",
    "speed_std_pxf", "direction_entropy", "directional_consistency",
    "flow_mean_magnitude", "flow_std_pxf", "track_count", "track_persistence",
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

    @property
    def fingerprint(self) -> str:
        return schema_fingerprint(self.names)


_DEFAULT = FeatureSchema()
