"""Tests for ByteTrack tracking wrapper and trajectory storage."""

import numpy as np
import pytest

from src.detection.detector import Detection
from src.tracking.tracker import CrowdTracker, TrackedPerson
from src.tracking.trajectories import TrackPoint, TrajectoryStore


def _det(x1, y1, x2, y2, conf=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls_id=0)


# ------------------------------------------------------------------ tracker
def test_ids_stable_for_moving_pair():
    tracker = CrowdTracker(frame_width=200, frame_height=100)
    ids_per_frame = []
    for f in range(5):
        dx = f * 2.0
        dets = [_det(10 + dx, 10, 30 + dx, 50), _det(50 + dx, 20, 70 + dx, 60)]
        persons = tracker.update(dets)
        assert len(persons) == 2
        ids_per_frame.append(sorted(p.track_id for p in persons))
    assert all(ids == ids_per_frame[0] for ids in ids_per_frame)
    assert len(set(ids_per_frame[0])) == 2


def test_empty_update_is_safe():
    tracker = CrowdTracker(frame_width=200, frame_height=100)
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert tracker.frame_count == 2


def test_tracked_person_fields():
    tracker = CrowdTracker(frame_width=200, frame_height=100)
    persons = tracker.update([_det(10, 10, 30, 50, conf=0.75)])
    p = persons[0]
    assert isinstance(p, TrackedPerson)
    assert p.track_id >= 1
    assert p.x1 == pytest.approx(10)
    assert p.conf == pytest.approx(0.75, abs=1e-3)


# -------------------------------------------------------------- trajectories
@pytest.fixture
def store():
    s = TrajectoryStore()
    # track 1: observed at frames 0..3, moving +2 px/frame in x
    for f in range(4):
        s.add_frame(f, [TrackedPerson(track_id=1, x1=10 + 2 * f, y1=10, x2=30 + 2 * f, y2=50, conf=0.9)])
    # track 2: only at frames 0 and 3 (gaps)
    s.add_frame(0, [TrackedPerson(track_id=2, x1=100, y1=10, x2=120, y2=50, conf=0.9)])
    s.add_frame(3, [TrackedPerson(track_id=2, x1=100, y1=10, x2=120, y2=50, conf=0.9)])
    return s


def test_track_ids(store):
    assert sorted(store.track_ids()) == [1, 2]


def test_frame_range(store):
    assert store.frame_range == (0, 3)
    empty = TrajectoryStore()
    assert empty.frame_range == (0, 0)


def test_trajectory_points_and_centers(store):
    traj = store.trajectory(1)
    assert traj.shape == (4, 3)  # frame, cx, cy
    np.testing.assert_allclose(traj[:, 0], [0, 1, 2, 3])
    np.testing.assert_allclose(traj[0], [0, 20, 30])  # frame0, cx=20, cy=30


def test_speeds_consecutive_frames_px_per_frame(store):
    speeds = store.speeds(1)
    assert speeds.shape == (3,)
    np.testing.assert_allclose(speeds, 2.0)  # +2 px in x per frame


def test_speeds_skip_gaps(store):
    # track 2 seen at frames 0 and 3 -> single speed over the gap?
    # by design only consecutive frames count -> no speed entries
    speeds = store.speeds(2)
    assert speeds.size == 0
    assert np.isnan(store.mean_speed(2))


def test_persistence(store):
    # total frames added: 0..3 (frame 0 twice counts once)
    assert store.total_frames == 4
    assert store.persistence(1) == pytest.approx(1.0)
    assert store.persistence(2) == pytest.approx(2 / 4)


def test_track_point_dataclass():
    pt = TrackPoint(frame_index=0, track_id=1, cx=20.0, cy=30.0,
                    x1=10, y1=10, x2=30, y2=50, conf=0.9)
    assert pt.track_id == 1
