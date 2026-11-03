"""HTTP API tests (FastAPI TestClient) + shared presentation helpers."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from src.detection.detector import Detection, DetectionResult
from src.inference import presentation
from src.inference.presentation import interpret_prediction, model_caveat

client = TestClient(app)


class FakeDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        h, w = image.shape[:2]
        return DetectionResult(
            detections=(Detection(x1=10, y1=10, x2=30, y2=40, conf=0.9, cls_id=0),),
            inference_time_s=0.0,
            model_name="fake",
            image_shape=(h, w),
        )


def _png_bytes(width=96, height=64) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------ endpoints
def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "crowdsense-ai"


def test_model_endpoint_reports_availability_and_caveat():
    response = client.get("/api/model")
    assert response.status_code == 200
    body = response.json()
    assert "available" in body
    assert "estimate" in body["caveat"].lower()
    if body["available"]:
        assert body["algorithm"]
        assert body["feature_names"]


def test_detect_image_returns_boxes_grid_and_static_features(monkeypatch):
    monkeypatch.setattr("src.detection.detector.PersonDetector", FakeDetector)
    response = client.post(
        "/api/detect-image",
        files={"file": ("frame.png", _png_bytes(), "image/png")},
        data={"conf": "0.05", "imgsz": "640"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert len(body["boxes"]) == 1
    assert body["boxes"][0]["confidence"] == pytest.approx(0.9)
    assert len(body["grid_counts"]) == body["grid_size"] == 4
    assert [f["name"] for f in body["static_features"]] == [
        "person_count",
        "density_per_mpx",
        "grid_concentration",
        "spread_x",
        "spread_y",
        "grid_entropy",
    ]
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert body["image"] == {"width": 96, "height": 64}
    assert "movement" in body["model_note"].lower()


def test_detect_frame_returns_detector_estimate(monkeypatch):
    monkeypatch.setattr("src.detection.detector.PersonDetector", FakeDetector)
    response = client.post(
        "/api/detect-frame",
        files={"frame": ("frame.png", _png_bytes(64, 64), "image/png")},
        data={"conf": "0.05", "imgsz": "320"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["metadata"]["model_name"] == "fake"


def test_detect_image_rejects_undecodable_upload():
    response = client.post(
        "/api/detect-image",
        files={"file": ("bad.png", b"not an image", "image/png")},
        data={"conf": "0.05", "imgsz": "640"},
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_detect_image_rejects_unsupported_imgsz():
    response = client.post(
        "/api/detect-image",
        files={"file": ("frame.png", _png_bytes(), "image/png")},
        data={"conf": "0.05", "imgsz": "999"},
    )
    assert response.status_code == 422


def test_detect_video_rejects_unsupported_extension():
    response = client.post(
        "/api/detect-video",
        files={"file": ("notes.txt", b"not a video", "text/plain")},
        data={"max_frames": "30"},
    )
    assert response.status_code == 400
    assert "unsupported video type" in response.json()["detail"].lower()


def test_detect_video_rejects_stride_larger_than_window():
    response = client.post(
        "/api/detect-video",
        files={"file": ("clip.mp4", b"not a video", "video/mp4")},
        data={"window": "30", "window_stride": "31"},
    )
    assert response.status_code == 422


def test_spa_deep_links_serve_index_when_built(project_root):
    index = project_root / "backend" / "static" / "index.html"
    if not index.is_file():
        pytest.skip("frontend build output absent (run npm run build)")
    for route in ("/", "/video", "/live-camera"):
        response = client.get(route)
        assert response.status_code == 200
        assert 'id="root"' in response.text


# ------------------------------------------------------------- presentation
def test_interpret_prediction_branches():
    cases = [
        (1, 0.9, "Anomaly likely"),
        (1, 0.6, "Possible anomaly"),
        (0, 0.45, "Probably normal"),
        (0, 0.1, "Normal"),
    ]
    for label, score, expected in cases:
        headline, note = interpret_prediction(label, score)
        assert headline == expected
        assert note


def test_interpret_prediction_boundaries():
    assert interpret_prediction(1, 0.8)[0] == "Anomaly likely"
    assert interpret_prediction(1, 0.79)[0] == "Possible anomaly"
    assert interpret_prediction(0, 0.3)[0] == "Probably normal"
    assert interpret_prediction(0, 0.29)[0] == "Normal"


def test_model_caveat_mentions_methodology_and_estimates():
    text = model_caveat()
    assert "Methodology" in text
    assert "estimate" in text.lower()


def test_find_model_path_returns_newest_bundle(tmp_path):
    older = tmp_path / "older.joblib"
    newer = tmp_path / "newer.joblib"
    older.write_bytes(b"x")
    newer.write_bytes(b"x")
    os.utime(older, (1_600_000_000, 1_600_000_000))
    os.utime(newer, (1_700_000_000, 1_700_000_000))
    assert presentation.find_model_path(tmp_path) == newer


def test_find_model_path_missing_dir_returns_none(tmp_path):
    assert presentation.find_model_path(tmp_path / "absent") is None
