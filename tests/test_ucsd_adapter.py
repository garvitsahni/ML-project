"""Tests for the UCSD Pedestrian Anomaly dataset adapter."""

from pathlib import Path

import numpy as np
import pytest

from src.datasets.ucsd_anomaly import (
    AnomalySequence,
    AnnotationError,
    MissingDatasetError,
    UcsdAdapter,
    parse_gt_frames,
)
from tests.fixtures import make_ucsd_fixture


# ---------------------------------------------------------------- GT .m parsing
def test_parse_gt_frames_simple_range():
    entries = parse_gt_frames("TestVideoFile{end+1}.gt_frame = [60:152];\n")
    assert len(entries) == 1
    assert entries[0] == frozenset(range(60, 153))


def test_parse_gt_frames_comma_separated_ranges():
    entries = parse_gt_frames("TestVideoFile{end+1}.gt_frame = [5:90, 140:200];\n")
    assert entries[0] == frozenset(list(range(5, 91)) + list(range(140, 201)))


def test_parse_gt_frames_multiple_entries_in_order():
    text = (
        "TestVideoFile = {};\n"
        "TestVideoFile{end+1}.gt_frame = [3:5];\n"
        "TestVideoFile{end+1}.gt_frame = [1:2, 5:6];\n"
    )
    entries = parse_gt_frames(text)
    assert len(entries) == 2
    assert entries[1] == frozenset({1, 2, 5, 6})


def test_parse_gt_frames_empty_text_returns_no_entries():
    assert parse_gt_frames("") == []
    assert parse_gt_frames("TestVideoFile = {};\n") == []


# ---------------------------------------------------------------- adapter
@pytest.fixture
def fixture_root(tmp_path) -> Path:
    return make_ucsd_fixture(tmp_path / "UCSD")


def test_missing_root_raises_missing_dataset(tmp_path):
    adapter = UcsdAdapter(root=tmp_path / "does_not_exist")
    with pytest.raises(MissingDatasetError) as exc:
        adapter.inspect()
    assert "does_not_exist" in str(exc.value)


def test_inspect_counts_images_and_annotations(fixture_root):
    report = UcsdAdapter(root=fixture_root).inspect()
    # 5+4 train + 10+6 test (ped1) + 3 train + 4 test (ped2)
    assert report.image_count == 5 + 4 + 10 + 6 + 3 + 4
    # anomalous frames: ped1 Test001 -> 3, Test002 -> 4, ped2 Test001 -> 3
    assert report.annotation_count == 3 + 4 + 3
    assert report.images_missing == []
    assert report.annotations_missing == []
    assert any("frame" in label for label in report.labels_available)


def test_inspect_reports_missing_gt_file(tmp_path):
    root = make_ucsd_fixture(tmp_path / "UCSD")
    (root / "UCSDped2" / "Test" / "UCSDped2.m").unlink()
    report = UcsdAdapter(root=root).inspect()
    assert any("UCSDped2.m" in a for a in report.annotations_missing)


def test_inspect_reports_gt_beyond_frame_count(tmp_path):
    root = make_ucsd_fixture(tmp_path / "UCSD")
    (root / "UCSDped1" / "Test" / "UCSDped1.m").write_text(
        "TestVideoFile{end+1}.gt_frame = [3:5];\n"
        "TestVideoFile{end+1}.gt_frame = [1:200];\n",  # Test002 only has 6 frames
        encoding="utf-8",
    )
    report = UcsdAdapter(root=root).inspect()
    assert any("Test002" in m for m in report.images_missing)


def test_iter_sequences_yields_all_splits(fixture_root):
    adapter = UcsdAdapter(root=fixture_root)
    seqs = list(adapter.iter_sequences())
    assert len(seqs) == 6
    train = [s for s in seqs if s.split == "train"]
    test = [s for s in seqs if s.split == "test"]
    assert len(train) == 3 and len(test) == 3


def test_train_sequences_are_all_normal_per_readme(fixture_root):
    adapter = UcsdAdapter(root=fixture_root)
    for seq in adapter.iter_sequences(split="train"):
        assert seq.frame_labels.sum() == 0, seq.sequence_id


def test_test_sequence_labels_match_gt(fixture_root):
    adapter = UcsdAdapter(root=fixture_root)
    seq = {s.sequence_id: s for s in adapter.iter_sequences(split="test")}
    s1 = seq["ped1_Test001"]
    assert s1.n_frames == 10
    # GT [3:5] = frames 3..5 (1-based) -> indices 2..4
    np.testing.assert_array_equal(
        s1.frame_labels, np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    )
    s2 = seq["ped1_Test002"]
    np.testing.assert_array_equal(s2.frame_labels, np.array([1, 1, 0, 0, 1, 1]))
    p2 = seq["ped2_Test001"]
    np.testing.assert_array_equal(p2.frame_labels, np.array([0, 1, 1, 1]))


def test_sequence_metadata_fields(fixture_root):
    seq = next(UcsdAdapter(root=fixture_root).iter_sequences(split="test"))
    assert isinstance(seq, AnomalySequence)
    assert seq.frames_dir.name.startswith("Test")
    assert len(seq.frame_files) == seq.n_frames
    assert seq.has_masks in (True, False)


def test_part_filter(fixture_root):
    seqs = list(UcsdAdapter(root=fixture_root).iter_sequences(part="ped2"))
    assert len(seqs) == 2
    assert all(s.part == "ped2" for s in seqs)


def test_annotation_count_mismatch_raises(fixture_root):
    # .m has 2 entries but ped1 has 2 test sequences -> ok by construction;
    # break it by deleting Test002 so entries (2) != sequences (1)
    import shutil

    shutil.rmtree(fixture_root / "UCSDped1" / "Test" / "Test002")
    adapter = UcsdAdapter(root=fixture_root)
    with pytest.raises(AnnotationError):
        list(adapter.iter_sequences(split="test"))
