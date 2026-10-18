"""Image loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Raised when an image file is missing, undecodable, or out of bounds."""


@dataclass(frozen=True)
class ImageMetadata:
    path: Path
    width: int
    height: int
    n_channels: int
    file_size: int


def validate_image(
    path: Path | str,
    *,
    min_dim: int = 32,
    max_dim: int = 8192,
) -> tuple[np.ndarray, ImageMetadata]:
    """Load a BGR image and validate size/decodability.

    Raises :class:`InvalidImageError` for missing files, undecodable
    content, or dimensions outside ``[min_dim, max_dim]`` per side.
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"Image file not found: {path}")
    file_size = path.stat().st_size
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError(f"Could not decode image: {path}")
    height, width = img.shape[:2]
    if min(width, height) < min_dim:
        raise InvalidImageError(
            f"Image too small: {width}x{height} (minimum {min_dim}px per side): {path}"
        )
    if max(width, height) > max_dim:
        raise InvalidImageError(
            f"Image too large: {width}x{height} (maximum {max_dim}px per side): {path}"
        )
    meta = ImageMetadata(
        path=path, width=width, height=height,
        n_channels=img.shape[2] if img.ndim == 3 else 1,
        file_size=file_size,
    )
    return img, meta
