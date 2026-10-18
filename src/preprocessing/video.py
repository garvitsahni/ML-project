"""Video file iteration (upload path) and single-frame reading.

Frame *sequences* (UCSD) are enumerated by the dataset adapter; this module
covers uploaded video files and reading individual frames from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class InvalidVideoError(ValueError):
    """Raised when a video file is missing or cannot be decoded."""


def read_frame(path: Path | str) -> np.ndarray:
    """Read a single image frame (BGR, 3-channel) from disk.

    Grayscale sources (e.g. UCSD ``.tif`` frames) are converted to BGR so
    downstream code always sees a 3-channel array. Handles both ``(H, W)``
    and ``(H, W, 1)`` results — after ultralytics/ByteTrack loads, ``imread``
    can return single-channel data with a trailing axis.
    """
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise InvalidVideoError(f"Could not read frame: {path}")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame


def iter_video_frames(
    video_path: Path | str,
    *,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` pairs from a video file.

    Raises :class:`InvalidVideoError` when the file cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise InvalidVideoError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoError(f"Could not open video: {video_path}")
    try:
        idx = 0
        while max_frames is None or idx < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()
