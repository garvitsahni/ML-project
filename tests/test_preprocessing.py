"""Tests for preprocessing: image validation, video iteration, camera."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing.camera import Camera, CameraUnavailableError
from src.preprocessing.image import ImageMetadata, InvalidImageError, validate_image
from src.preprocessing.video import InvalidVideoError, iter_video_frames, read_frame


# ------------------------------------------------------------------ images
def test_validate_image_returns_metadata_and_pixels(tmp_image):
    img, meta = validate_image(tmp_image)
    assert img.shape == (48, 64, 3)
    assert isinstance(meta, ImageMetadata)
    assert (meta.width, meta.height) == (64, 48)
    assert meta.n_channels == 3
    assert meta.file_size > 0


def test_validate_image_missing_file_raises(tmp_path):
    with pytest.raises(InvalidImageError):
        validate_image(tmp_path / "nope.png")


def test_validate_image_corrupt_file_raises(tmp_path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not an image at all")
    with pytest.raises(InvalidImageError):
        validate_image(bad)


def test_validate_image_too_small_raises(tmp_path):
    p = tmp_path / "tiny.png"
    cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        validate_image(p, min_dim=32)


def test_validate_image_too_large_raises(tmp_path):
    p = tmp_path / "wide.png"
    cv2.imwrite(str(p), np.zeros((16, 9000, 3), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        validate_image(p, max_dim=8192)


# ------------------------------------------------------------------- video
def _write_video(path: Path, n: int = 12, size=(64, 48)) -> Path:
    vw = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size
    )
    assert vw.isOpened(), "VideoWriter failed to open"
    for i in range(n):
        frame = np.full((size[1], size[0], 3), i * 20 % 255, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return path


def test_iter_video_frames_counts_and_order(tmp_path):
    vp = _write_video(tmp_path / "clip.mp4", n=12)
    frames = list(iter_video_frames(vp))
    assert len(frames) == 12
    assert [idx for idx, _ in frames] == list(range(12))
    assert frames[0][1].shape == (48, 64, 3)


def test_iter_video_frames_max_frames(tmp_path):
    vp = _write_video(tmp_path / "clip.mp4", n=12)
    frames = list(iter_video_frames(vp, max_frames=5))
    assert len(frames) == 5


def test_iter_video_frames_missing_raises(tmp_path):
    with pytest.raises(InvalidVideoError):
        list(iter_video_frames(tmp_path / "missing.mp4"))


def test_iter_video_frames_corrupt_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x01garbage")
    with pytest.raises(InvalidVideoError):
        list(iter_video_frames(bad))


def test_read_frame_from_sequence(tmp_path):
    p = tmp_path / "001.tif"
    cv2.imwrite(str(p), np.full((10, 12), 200, dtype=np.uint8))
    frame = read_frame(p)
    assert frame.shape[:2] == (10, 12)


# ------------------------------------------------------------------ camera
def test_camera_missing_device_raises():
    cam = Camera(device=99)
    with pytest.raises(CameraUnavailableError):
        cam.open()


def test_camera_reads_from_injected_capture(tmp_path):
    """Camera logic tested with a file-backed capture (no hardware needed)."""
    vp = _write_video(tmp_path / "feed.mp4", n=6)
    cam = Camera(device=0)
    real_cap = cv2.VideoCapture

    def fake_capture(device):
        return real_cap(str(vp)) if device == 0 else real_cap(device)

    cv2.VideoCapture = fake_capture
    try:
        with cam:
            ok, frame = cam.read()
            assert ok and frame is not None
            assert cam.is_open
        assert not cam.is_open
    finally:
        cv2.VideoCapture = real_cap


def test_camera_unopened_read_fails_cleanly():
    cam = Camera(device=99)
    ok, frame = cam.read()
    assert ok is False and frame is None
