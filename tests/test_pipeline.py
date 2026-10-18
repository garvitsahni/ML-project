"""Tests for the sequence->window feature-table pipeline (FakeDetector, fast)."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.datasets.ucsd_anomaly import UcsdAdapter
from src.detection.detector import Detection, DetectionResult
from src.features.pipeline import WindowConfig, build_feature_table, process_sequence
from src.features.schema import FeatureSchema
from tests.fixtures import make_ucsd_fixture


@dataclass
class MovingFakeDetector:
    """One person moving +2 px/frame in x, starting at x1=10."""

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        # use image brightness as frame proxy: fixture frames are i*10 % 255
        f = int(round(image.mean() / 10.0)) % 255
        x1 = 10 + 2.0 * f
        det = Detection(x1=x1, y1=10, x2=x1 + 20, y2=50, conf=0.9, cls_id=0)
        return DetectionResult((det,), 0.0, "fake", (h, w))


@pytest.fixture
def fixture_root(tmp_path):
    return make_ucsd_fixture(tmp_path / "UCSD")


def _seq(root, seq_id):
    return {s.sequence_id: s for s in UcsdAdapter(root=root).iter_sequences()}[seq_id]


def test_process_sequence_window_count(fixture_root):
    seq = _seq(fixture_root, "ped1_Test001")  # 10 frames
    rows = process_sequence(seq, detector=MovingFakeDetector(),
                            config=WindowConfig(window=4, stride=2, grid_size=2))
    assert len(rows) == 4  # starts 0,2,4,6


def test_rows_schema_columns(fixture_root):
    seq = _seq(fixture_root, "ped1_Test001")
    rows = process_sequence(seq, detector=MovingFakeDetector(),
                            config=WindowConfig(window=4, stride=2, grid_size=2))
    schema = FeatureSchema.default()
    expected_meta = {"sequence_id", "part", "split", "window_start", "window_end", "label"}
    assert expected_meta.issubset(rows[0].keys())
    assert [c for c in rows[0] if c not in expected_meta] == schema.names


def test_window_labels_from_gt(fixture_root):
    # GT frames 3-5 anomalous -> windows (0-3)=1, (2-5)=1, (4-7)=1, (6-9)=0
    seq = _seq(fixture_root, "ped1_Test001")
    rows = process_sequence(seq, detector=MovingFakeDetector(),
                            config=WindowConfig(window=4, stride=2, grid_size=2))
    assert [r["label"] for r in rows] == [1, 1, 1, 0]


def test_train_windows_all_normal(fixture_root):
    seq = _seq(fixture_root, "ped1_Train001")  # 5 frames, all normal
    rows = process_sequence(seq, detector=MovingFakeDetector(),
                            config=WindowConfig(window=3, stride=2, grid_size=2))
    assert rows and all(r["label"] == 0 for r in rows)


def test_mean_speed_is_two_px_per_frame(fixture_root):
    # fixture frames are uniform gray i*10%255 -> fake detector frame proxy
    seq = _seq(fixture_root, "ped1_Test001")
    rows = process_sequence(seq, detector=MovingFakeDetector(),
                            config=WindowConfig(window=4, stride=2, grid_size=2))
    # detector moves the box +2px/frame; tracked centers are Kalman-smoothed,
    # so speed converges near 2.0 but not exactly (allow lag tolerance)
    for r in rows:
        assert r["mean_speed_pxf"] == pytest.approx(2.0, abs=0.15)
        assert r["person_count"] == pytest.approx(1.0)


def test_empty_sequence_yields_no_rows():
    from pathlib import Path

    from src.datasets.ucsd_anomaly import AnomalySequence

    empty = AnomalySequence(
        sequence_id="empty", part="ped1", split="test",
        frames_dir=Path("."), frame_files=(), n_frames=0,
        frame_labels=np.zeros(0, dtype=np.int64), has_masks=False,
    )
    rows = process_sequence(empty, detector=MovingFakeDetector(),
                            config=WindowConfig(window=4, stride=2, grid_size=2))
    assert rows == []


def test_build_feature_table_all_sequences(fixture_root):
    adapter = UcsdAdapter(root=fixture_root)
    df = build_feature_table(adapter, detector=MovingFakeDetector(),
                             config=WindowConfig(window=3, stride=3, grid_size=2))
    # sequences with >=3 frames: ped1 Train001(5), Train002(4), Test001(10), Test002(6),
    # ped2 Train001(3), Test001(4) -> all produce >=1 window
    assert len(df) == 1 + 1 + 3 + 2 + 1 + 1  # (n-window)//stride + 1 per sequence
    assert set(df["split"]) == {"train", "test"}
    assert df["label"].isin([0, 1]).all()
    assert list(FeatureSchema.default().names) == [
        c for c in df.columns if c not in
        {"sequence_id", "part", "split", "window_start", "window_end", "label"}
    ]
