"""Tests for YOLO person detection and image analysis.

Uses the real yolov8n.pt weights (CPU) on synthetic images plus an injected
fake detector for exact-count logic.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.detection.analyze import ImageAnalysis, analyze_frame, analyze_image
from src.detection.detector import (
    Detection,
    DetectionResult,
    ModelNotFoundError,
    PersonDetector,
    draw_detections,
)

WEIGHTS = PROJECT_ROOT / "yolov8n.pt"


@dataclass
class FakeDetector:
    """Deterministic stand-in returning hand-set boxes."""

    boxes: tuple

    def detect(self, image: np.ndarray) -> DetectionResult:
        dets = tuple(
            Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=0.9, cls_id=0)
            for (x1, y1, x2, y2) in self.boxes
        )
        return DetectionResult(
            detections=dets, inference_time_s=0.0, model_name="fake", image_shape=image.shape
        )


def synthetic_scene(h=480, w=640):
    """A non-trivial image (not uniform noise) so YOLO runs stably."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return img


# ---------------------------------------------------------------- detector
def test_model_missing_raises():
    with pytest.raises(ModelNotFoundError):
        PersonDetector(model_path=PROJECT_ROOT / "nonexistent.pt").detect(synthetic_scene(64, 64))


def test_detect_smoke_real_model():
    det = PersonDetector(model_path=WEIGHTS, conf=0.25)
    result = det.detect(synthetic_scene())
    assert isinstance(result, DetectionResult)
    assert result.model_name == WEIGHTS.name
    assert result.inference_time_s > 0
    h, w = synthetic_scene().shape[:2]
    for d in result.detections:
        assert d.cls_id == 0  # person-only
        assert 0.0 <= d.conf <= 1.0
        assert 0 <= d.x1 <= d.x2 <= w and 0 <= d.y1 <= d.y2 <= h


def test_count_helper_uses_fake():
    fake = FakeDetector(boxes=((10, 10, 30, 50), (50, 20, 70, 60)))
    result = fake.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(result.detections) == 2


# ----------------------------------------------------------------- drawing
def test_draw_detections_does_not_mutate_input():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    det = Detection(x1=10, y1=10, x2=50, y2=60, conf=0.9, cls_id=0)
    out = draw_detections(img, (det,))
    assert out.shape == img.shape
    assert np.array_equal(img, np.zeros_like(img))  # input untouched
    assert not np.array_equal(out, img)  # something was drawn


# ------------------------------------------------------------------ analyze
def test_analyze_frame_with_fake_boxes():
    fake = FakeDetector(boxes=((10, 10, 30, 50), (50, 20, 70, 60), (120, 40, 140, 80)))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    analysis = analyze_frame(frame, detector=fake, grid_size=4)
    assert isinstance(analysis, ImageAnalysis)
    assert analysis.count == 3
    assert analysis.density_map.shape == (100, 200)
    assert analysis.grid_counts.shape == (4, 4)
    assert analysis.grid_counts.sum() == 3


def test_analyze_image_counts_match_detector(tmp_image):
    fake = FakeDetector(boxes=((10, 10, 30, 30),))
    analysis = analyze_image(tmp_image, detector=fake)
    assert analysis.count == 1
    assert (analysis.meta.width, analysis.meta.height) == (64, 48)


def test_analyze_frame_empty_scene():
    fake = FakeDetector(boxes=())
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    analysis = analyze_frame(frame, detector=fake, grid_size=2)
    assert analysis.count == 0
    assert analysis.grid_counts.sum() == 0
    assert analysis.density_map.max() == 0


def test_density_grid_counts_centers():
    # one box whose center falls in cell (row 0, col 0) of a 4x4 grid on 100x200
    fake = FakeDetector(boxes=((0, 0, 20, 20),))  # center (10,10) -> cell (0,0)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    analysis = analyze_frame(frame, detector=fake, grid_size=4)
    assert analysis.grid_counts[0, 0] == 1
    assert analysis.grid_counts.sum() == 1
