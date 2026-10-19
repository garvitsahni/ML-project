"""Shared pytest fixtures for CrowdSense AI.

Synthetic fixtures are used for UNIT TESTS ONLY and never to report
research performance (see README "Limitations").
"""

import os
import sys
from pathlib import Path

# Pin native thread pools to 1 BEFORE numpy/torch/xgboost/OpenCV import.
# Mixed OpenMP runtimes (BLAS + torch + xgboost) cause intermittent
# access-violation crashes inside numpy/matplotlib/cv2 on Windows.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
import pytest

# Make the project root importable regardless of pytest invocation dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def tmp_image(tmp_path):
    """A valid 64x48 BGR PNG written to disk."""
    import cv2

    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[10:30, 10:30] = (0, 255, 0)
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def box_geometry():
    """Fixed detections with hand-computable geometry (for feature tests).

    Two tracks, frame shape (100, 200) = (h, w).
    """
    return {
        "frame_shape": (100, 200),
        # detections at one instant: (x1, y1, x2, y2, conf)
        "dets": np.array(
            [
                [10.0, 10.0, 30.0, 50.0, 0.9],
                [50.0, 20.0, 70.0, 60.0, 0.8],
                [120.0, 40.0, 140.0, 80.0, 0.7],
                [150.0, 10.0, 170.0, 50.0, 0.6],
            ],
            dtype=np.float64,
        ),
    }
