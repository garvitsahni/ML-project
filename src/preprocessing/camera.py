"""Webcam capture with strict cleanup semantics.

All camera failures raise :class:`CameraUnavailableError` (no silent
placeholders). On Windows, always use the context manager or ``finally`` so
the device is released.
"""

from __future__ import annotations

import numpy as np


class CameraUnavailableError(RuntimeError):
    """Raised when a camera device cannot be opened."""


class Camera:
    """Thin wrapper over ``cv2.VideoCapture`` for webcam use."""

    def __init__(self, device: int = 0):
        self.device = device
        self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> "Camera":
        if self.is_open:
            return self
        import cv2

        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(
                f"Camera device {self.device} is not available. "
                "Connect a camera or choose another device index."
            )
        self._cap = cap
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Grab one frame; returns ``(False, None)`` if not open or empty."""
        if not self.is_open:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
