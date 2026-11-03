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
    mean, std, _ = optical_flow_stats(img, img.copy())
    assert mean == pytest.approx(0.0, abs=1e-6)


def test_optical_flow_translated_image_positive():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    shifted = np.roll(img, shift=3, axis=1)  # move right 3px
    mean, std, _ = optical_flow_stats(img, shifted)
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


# ------------------------------------------------- schema v2 movement features
def test_schema_v2_movement_features_present():
    schema = FeatureSchema.default()
    v2 = ["nn_distance_mean", "nn_distance_std", "abrupt_stop_fraction",
          "direction_reversal_rate", "fast_mover_fraction", "speed_p90",
          "track_turnover_rate"]
    for name in v2:
        assert name in schema.names, name
    for spec in schema.specs:
        if spec.name in v2:
            assert spec.group == "movement"


def test_nearest_neighbor_stats_nominal():
    from src.features.movement import nearest_neighbor_stats

    centers_by_frame = {
        0: np.array([[0.0, 0.0], [30.0, 40.0]]),   # NN dist 50 per center
        1: np.array([[0.0, 0.0], [60.0, 80.0]]),   # NN dist 100 per center
    }
    mean, std = nearest_neighbor_stats(TrajectoryStore(), [], centers_by_frame,
                                       image_diagonal=100.0)
    assert mean == pytest.approx(0.75)   # mean([50, 100]) / 100
    assert std == pytest.approx(0.25)    # std([50, 100]) / 100


def test_nearest_neighbor_stats_degenerate():
    from src.features.movement import nearest_neighbor_stats

    empty = nearest_neighbor_stats(TrajectoryStore(), [], {}, image_diagonal=100.0)
    assert empty == (0.0, 0.0)
    single = nearest_neighbor_stats(TrajectoryStore(), [],
                                    {0: np.array([[5.0, 5.0]])}, image_diagonal=100.0)
    assert single == (0.0, 0.0)


def test_abrupt_stop_fraction_nominal():
    from src.features.movement import abrupt_stop_fraction

    # track 1: 10,10,1,0.5 -> decelerates hard (last <= 0.25 * max, max >= 2 * median)
    store = TrajectoryStore()
    store.add_frame(0, [TrackedPerson(track_id=1, x1=0, y1=0, x2=20, y2=20, conf=0.9)])
    store.add_frame(1, [TrackedPerson(track_id=1, x1=10, y1=0, x2=30, y2=20, conf=0.9)])
    store.add_frame(2, [TrackedPerson(track_id=1, x1=20, y1=0, x2=40, y2=20, conf=0.9)])
    store.add_frame(3, [TrackedPerson(track_id=1, x1=21, y1=0, x2=41, y2=20, conf=0.9)])
    store.add_frame(4, [TrackedPerson(track_id=1, x1=21.5, y1=0, x2=41.5, y2=20, conf=0.9)])
    # track 2: steady 1 px/frame -> not a stop
    for f in range(5):
        store.add_frame(f, [TrackedPerson(track_id=2, x1=100.0 + f, y1=0,
                                          x2=120.0 + f, y2=20, conf=0.9)])
    frac = abrupt_stop_fraction(store, [1, 2], median_window_speed=1.0)
    assert frac == pytest.approx(0.5)  # 1 of 2 tracks


def test_abrupt_stop_fraction_degenerate():
    from src.features.movement import abrupt_stop_fraction

    assert abrupt_stop_fraction(TrajectoryStore(), [], 1.0) == 0.0
    store = TrajectoryStore()  # single track, 1 speed sample (< 3)
    store.add_frame(0, [TrackedPerson(track_id=1, x1=0, y1=0, x2=20, y2=20, conf=0.9)])
    store.add_frame(1, [TrackedPerson(track_id=1, x1=100, y1=0, x2=120, y2=20, conf=0.9)])
    assert abrupt_stop_fraction(store, [1], median_window_speed=1.0) == 0.0


def test_direction_reversal_rate_nominal():
    from src.features.movement import direction_reversal_rate

    # 6 frames: right,right,right,left,left -> unit steps R,R,R,L,L
    # step pairs: 4; reversals (dot < -0.36): only R->L at index 2 -> 1/4
    xs = [0.0, 2.0, 4.0, 6.0, 4.0, 2.0]
    steps = [(f, xs[f], 0.0, xs[f] + 20, 20.0) for f in range(6)]
    store = _store_from_tracks({1: steps})
    assert direction_reversal_rate(store, [1]) == pytest.approx(0.25)


def test_direction_reversal_rate_degenerate():
    from src.features.movement import direction_reversal_rate

    assert direction_reversal_rate(TrajectoryStore(), []) == 0.0
    store = _store_from_tracks({1: [(0, 0.0, 0.0, 20.0, 20.0)]})
    assert direction_reversal_rate(store, [1]) == 0.0


def test_fast_mover_fraction_nominal():
    from src.features.movement import fast_mover_fraction

    store = TrajectoryStore()
    # track 1: 10 px/frame (fast), track 2: 1 px/frame (not), track 3: 4 px/frame (fast)
    for tid, speed in ((1, 10.0), (2, 1.0), (3, 4.0)):
        for f in range(4):
            x = speed * f
            store.add_frame(f, [TrackedPerson(track_id=tid, x1=x, y1=float(tid) * 30,
                                              x2=x + 20, y2=float(tid) * 30 + 20, conf=0.9)])
    frac = fast_mover_fraction(store, [1, 2, 3], median_window_speed=1.0)
    assert frac == pytest.approx(2 / 3)  # tracks 1 and 3 exceed 3x median


def test_fast_mover_fraction_degenerate():
    from src.features.movement import fast_mover_fraction

    assert fast_mover_fraction(TrajectoryStore(), [], 1.0) == 0.0
    store = TrajectoryStore()  # one track, no speed samples
    store.add_frame(0, [TrackedPerson(track_id=1, x1=0, y1=0, x2=20, y2=20, conf=0.9)])
    assert fast_mover_fraction(store, [1], 1.0) == 0.0


def test_speed_p90_nominal_and_degenerate():
    from src.features.movement import speed_p90

    steps = [(f, float(f), 0.0, float(f) + 20, 20.0) for f in range(5)]  # speeds all 1
    store = _store_from_tracks({1: steps})
    assert speed_p90(store, [1]) == pytest.approx(1.0)
    assert speed_p90(TrajectoryStore(), []) == 0.0


def test_track_turnover_rate_nominal():
    from src.features.movement import track_turnover_rate

    store = TrajectoryStore()
    # track 1: full window 0..9 (no birth/death); track 2: born at 3; track 3: dies at 5
    for f in range(10):
        store.add_frame(f, [TrackedPerson(track_id=1, x1=float(f), y1=0,
                                          x2=float(f) + 20, y2=20, conf=0.9)])
    for f in range(3, 10):
        store.add_frame(f, [TrackedPerson(track_id=2, x1=100.0, y1=30,
                                          x2=120.0, y2=50, conf=0.9)])
    for f in range(0, 6):
        store.add_frame(f, [TrackedPerson(track_id=3, x1=100.0, y1=60,
                                          x2=120.0, y2=80, conf=0.9)])
    rate = track_turnover_rate(store, [1, 2, 3], window_start=0, window_end=10)
    assert rate == pytest.approx(2 / 10)  # 1 birth + 1 death over 10 frames


def test_track_turnover_rate_degenerate():
    from src.features.movement import track_turnover_rate

    assert track_turnover_rate(TrajectoryStore(), [], 0, 10) == 0.0


# ------------------------------------------------- flow-direction inconsistency
def test_flow_dir_inconsistency_uniform_translation_near_zero():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    shifted = np.roll(img, shift=3, axis=1)
    mean, std, dir_inc = optical_flow_stats(img, shifted)
    assert dir_inc == pytest.approx(0.0, abs=0.15)  # all flow points one way


def test_flow_dir_inconsistency_static_is_zero():
    img = np.full((64, 64), 128, dtype=np.uint8)
    _, _, dir_inc = optical_flow_stats(img, img.copy())
    assert dir_inc == 0.0  # no motion pixels -> 0 by definition


def test_schema_v2_includes_flow_direction_inconsistency():
    schema = FeatureSchema.default()
    assert "flow_direction_inconsistency" in schema.names
    spec = next(s for s in schema.specs if s.name == "flow_direction_inconsistency")
    assert spec.group == "movement"


def test_extract_window_features_flow_dir_mean():
    schema = FeatureSchema.default()
    signals = [
        FrameSignals(frame_index=f, count=1,
                     grid_counts=np.array([[1, 0], [0, 0]]),
                     centers=np.array([[20.0, 30.0]]),
                     flow_mean=0.0, flow_std=0.0, flow_dir_inconsistency=v)
        for f, v in enumerate([0.2, 0.4])
    ]
    feats = extract_window_features(signals, TrajectoryStore(),
                                    schema=schema, frame_shape=FRAME_SHAPE)
    assert feats["flow_direction_inconsistency"] == pytest.approx(0.3)


def test_extract_window_features_wires_v2_movement_features():
    """Regression: the 7 schema-v2 features must be COMPUTED by the window
    aggregation (they once sat at their initialized 0.0 in the real table)."""
    schema = FeatureSchema.default()
    store = TrajectoryStore()
    signals = []

    def _persons(f):
        t1_x = float(f)                       # slow steady, speed 1
        t2_x = 10.0 + 10.0 * f                # fast mover, speed 10
        t3_x = [0.0, 4.0, 8.0, 12.0, 12.5, 12.75][f]  # abrupt stop
        t4_x = [0.0, 2.0, 4.0, 3.0, 2.0, 1.0][f]      # direction reversal
        out = [
            TrackedPerson(track_id=1, x1=t1_x, y1=0, x2=t1_x + 20, y2=20, conf=0.9),
            TrackedPerson(track_id=2, x1=t2_x, y1=25, x2=t2_x + 20, y2=45, conf=0.9),
            TrackedPerson(track_id=3, x1=t3_x, y1=50, x2=t3_x + 20, y2=70, conf=0.9),
            TrackedPerson(track_id=4, x1=t4_x, y1=75, x2=t4_x + 20, y2=95, conf=0.9),
        ]
        if f >= 3:                            # track 5 born mid-window -> turnover
            out.append(TrackedPerson(track_id=5, x1=180.0, y1=0, x2=200.0, y2=20, conf=0.9))
        return out

    for f in range(6):
        persons = _persons(f)
        store.add_frame(f, persons)
        centers = np.array([p.center for p in persons], dtype=float).reshape(-1, 2)
        signals.append(FrameSignals(
            frame_index=f, count=len(persons),
            grid_counts=np.array([[1, 1], [1, 1]]), centers=centers,
            flow_mean=1.0, flow_std=0.5, flow_dir_inconsistency=0.3,
        ))

    feats = extract_window_features(signals, store, schema=schema,
                                    frame_shape=FRAME_SHAPE)
    assert feats["nn_distance_mean"] > 0      # 4+ spread-out centers per frame
    assert feats["nn_distance_std"] > 0       # distances change as tracks move
    assert feats["abrupt_stop_fraction"] > 0  # track 3 decelerates hard
    assert feats["direction_reversal_rate"] > 0  # track 4 flips direction
    assert feats["fast_mover_fraction"] > 0   # track 2 is >3x median speed
    assert feats["speed_p90"] > 1.5           # pooled p90 well above median
    assert feats["track_turnover_rate"] > 0   # track 5 born mid-window
