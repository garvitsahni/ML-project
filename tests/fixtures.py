"""Synthetic fixtures for UNIT TESTS ONLY.

Nothing produced here is ever used to report research performance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_tif_frames(frames_dir: Path, n: int, size: tuple[int, int] = (8, 8)) -> list[Path]:
    """Write ``n`` small grayscale TIFF frames named 001.tif, 002.tif, ..."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        img = np.full(size, i * 10 % 255, dtype=np.uint8)
        p = frames_dir / f"{i:03d}.tif"
        assert cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def make_ucsd_fixture(root: Path) -> Path:
    """Create a miniature UCSD v1p2-style layout under ``root``.

    Layout mirrors the real dataset:
      UCSDped1/Train/Train001/*.tif
      UCSDped1/Test/Test001/*.tif, Test001_gt/*.bmp
      UCSDped1/Test/UCSDped1.m          <- frame-level gt ranges (1-based)
      UCSDped2/... (same shape, smaller)
    """
    root.mkdir(parents=True, exist_ok=True)

    write_tif_frames(root / "UCSDped1" / "Train" / "Train001", 5)
    write_tif_frames(root / "UCSDped1" / "Train" / "Train002", 4)
    write_tif_frames(root / "UCSDped1" / "Test" / "Test001", 10)
    write_tif_frames(root / "UCSDped1" / "Test" / "Test002", 6)
    # pixel-level mask folder for Test001 (binary bmp)
    gt_dir = root / "UCSDped1" / "Test" / "Test001_gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(gt_dir / "003.bmp"), np.full((8, 8), 255, dtype=np.uint8))

    (root / "UCSDped1" / "Test" / "UCSDped1.m").write_text(
        "TestVideoFile = {};\n"
        "TestVideoFile{end+1}.gt_frame = [3:5];\n"
        "TestVideoFile{end+1}.gt_frame = [1:2, 5:6];\n",
        encoding="utf-8",
    )

    write_tif_frames(root / "UCSDped2" / "Train" / "Train001", 3)
    write_tif_frames(root / "UCSDped2" / "Test" / "Test001", 4)
    (root / "UCSDped2" / "Test" / "UCSDped2.m").write_text(
        "TestVideoFile = {};\n"
        "TestVideoFile{end+1}.gt_frame = [2:4];\n",
        encoding="utf-8",
    )
    return root
