"""Tests for feature schema, static features, movement features, windows."""

import math

import numpy as np
import pytest

from src.detection.detector import Detection
from src.features.extract import (
    FrameSignals,
    extract_window_features,
    optical_flow_stats,
)
from src.features.movement import (
    directional_consistency,
    direction_entropy,
    movement_speeds_from_trajectories,
)
from src.features.schema import FeatureSchema
from src.features.static import static_features
from src.tracking.tracker import TrackedPerson
from src.tracking.trajectories import TrajectoryStore

FRAME_SHAPE = (100, 200)  # h, w


# ------------------------------------------------------------------- schema
def test_schema_names_unique_and_ordered():
    schema = FeatureSchema.default()
    names = schema.names
    assert len(names) == len(set(names))
    assert names == schema.names  # stable across calls
    assert "person_count" in names
    assert "mean_speed_pxf" in names


def test_schema_specs_have_unit_and_limitation():
    for spec in FeatureSchema.default().specs:
        assert spec.unit, spec.name
        assert spec.limitation, spec.name
        assert spec.group in ("static", "movement")


def test_schema_names_for_groups():
    schema = FeatureSchema.default()
    static_only = schema.names_for({"static"})
    assert static_only and "mean_speed_pxf" not in static_only
    both = schema.names_for({"static", "movement"})
    assert both == schema.names


# -------------------------------------------------------------- static feats
def test_static_features_hand_computed(box_geometry):
    centers = [(20.0, 30.0), (60.0, 40.0), (130.0, 60.0), (160.0, 30.0)]
    grid = np.array([[2, 2], [0, 0]])
    feats = static_features(count=4, grid_counts=grid, centers=np.array(centers),
                            frame_shape=FRAME_SHAPE, grid_size=2)
    assert feats["person_count"] == 4.0
    assert feats["density_per_mpx"] == pytest.approx(4.0 / (100 * 200 / 1e6))
    assert feats["grid_concentration"] == pytest.approx(0.5)  # 2/4
    # spread_x = std([20,60,130,160]) / 200
    assert feats["spread_x"] == pytest.approx(np.std([20, 60, 130, 160]) / 200)
    assert feats["spread_y"] == pytest.approx(np.std([30, 40, 60, 30]) / 100)
    # entropy of [0.5, 0.5, 0, 0] normalized by ln(4) -> ln2/ln4 = 0.5
    assert feats["grid_entropy"] == pytest.approx(0.5)


def test_static_features_empty_scene():
    feats = static_features(count=0, grid_counts=np.zeros((2, 2), dtype=int),
                            centers=np.zeros((0, 2)), frame_shape=FRAME_SHAPE, grid_size=2)
    assert feats["person_count"] == 0.0
    assert feats["grid_concentration"] == 0.0
    assert feats["spread_x"] == 0.0
    assert feats["grid_entropy"] == 0.0


# ------------------------------------------------------------- movement bits
def _store_from_tracks(track_steps):
    """track_steps: {track_id: [(frame, x1, y1, x2, y2), ...]}"""
    store = TrajectoryStore()
    for tid, steps in track_steps.items():
        for f, x1, y1, x2, y2 in steps:
            store.add_frame(f, [TrackedPerson(track_id=tid, x1=x1, y1=y1, x2=x2, y2=y2, conf=0.9)])
    return store


def test_speeds_and_direction_consistent_right():
    # one track moving +2 px/frame in x
    steps = [(f, 10 + 2 * f, 10, 30 + 2 * f, 50) for f in range(5)]
    store = _store_from_tracks({1: steps})
    speeds = movement_speeds_from_trajectories(store, [1])
    assert speeds.size == 4
    np.testing.assert_allclose(speeds, 2.0)
    unit = directional_consistency(store, [1])
    assert unit == pytest.approx(1.0)  # all rightward


def test_directional_consistency_opposite_flows_cancels():
    right = [(f, 10 + 2 * f, 10, 30 + 2 * f, 50) for f in range(5)]
    left = [(f, 100 - 2 * f, 10, 120 - 2 * f, 50) for f in range(5)]
    store = _store_from_tracks({1: right, 2: left})
    assert directional_consistency(store, [1, 2]) == pytest.approx(0.0, abs=1e-9)
    # 2 occupied bins of 8: H = ln2, normalized by ln8 -> 1/3
    assert direction_entropy(store, [1, 2]) == pytest.approx(math.log(2) / math.log(8))


def test_direction_entropy_uniform_is_high():
    # 8 tracks each moving in a different direction -> max-ish entropy
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    store = TrajectoryStore()
    for i, (dx, dy) in enumerate(dirs):
        for f in range(4):
            x = 10.0 * (i + 1) + dx * f
            y = 10.0 + dy * f
            store.add_frame(f, [TrackedPerson(track_id=i + 1, x1=x, y1=y, x2=x + 20, y2=y + 20, conf=0.9)])
    h = direction_entropy(store, [1, 2, 3, 4, 5, 6, 7, 8])
    assert h > 0.9
    assert h <= 1.0


# ---------------------------------------------------------------- optical flow
def test_optical_flow_static_image_near_zero():
    img = np.full((64, 64), 128, dtype=np.uint8)
    mean, std = optical_flow_stats(img, img.copy())
    assert mean == pytest.approx(0.0, abs=1e-6)


def test_optical_flow_translated_image_positive():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    shifted = np.roll(img, shift=3, axis=1)  # move right 3px
    mean, std = optical_flow_stats(img, shifted)
    assert mean > 0.5  # a constant 3px shift should register clear motion


# ----------------------------------------------------------- window assembly
def test_extract_window_features_keys_and_values():
    schema = FeatureSchema.default()
    # 4 frames, count 4, moving right 2 px/frame, no optical flow
    signals = [
        FrameSignals(frame_index=f, count=4,
                     grid_counts=np.array([[2, 2], [0, 0]]),
                     centers=np.array([[20 + 2 * f, 30], [60 + 2 * f, 40],
                                       [130 + 2 * f, 60], [160 + 2 * f, 30]], dtype=float),
                     flow_mean=0.0, flow_std=0.0)
        for f in range(4)
    ]
    store = TrajectoryStore()
    for f in range(4):
        for cx0 in (20, 60, 130, 160):
            x1 = cx0 + 2 * f - 10
            store.add_frame(f, [TrackedPerson(track_id=cx0, x1=x1, y1=10, x2=x1 + 20, y2=50, conf=0.9)])
    feats = extract_window_features(signals, store, schema=schema, frame_shape=FRAME_SHAPE)
    assert list(feats.keys()) == schema.names
    assert feats["person_count"] == 4.0
    assert feats["mean_speed_pxf"] == pytest.approx(2.0)
    assert feats["speed_std_pxf"] == pytest.approx(0.0, abs=1e-9)
    assert feats["directional_consistency"] == pytest.approx(1.0)
    assert feats["track_count"] == 4.0
    assert feats["density_delta"] == pytest.approx(0.0)
    assert feats["flow_mean_magnitude"] == 0.0


def test_extract_window_features_empty_signals():
    schema = FeatureSchema.default()
    feats = extract_window_features([], TrajectoryStore(), schema=schema, frame_shape=FRAME_SHAPE)
    assert list(feats.keys()) == schema.names
    assert all(v == 0.0 for v in feats.values())
