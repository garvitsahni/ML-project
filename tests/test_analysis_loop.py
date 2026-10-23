"""FrameAnalyzer shared-loop tests (fake detector; real tracker, flow, features).

The analyzer encapsulates the detect -> track -> flow -> window-features loop
that the Video and Live Camera pages previously duplicated inline.
"""

import numpy as np

from src.detection.detector import Detection, DetectionResult
from src.ui.analysis_loop import FrameAnalyzer


class FakeDetector:
    """Deterministic single-person detector (moving +2 px/frame in x)."""

    def __init__(self, offset_step: float = 2.0):
        self._offset = 0.0
        self._step = offset_step

    def detect(self, frame):
        h, w = frame.shape[:2]
        x = 10.0 + self._offset
        self._offset += self._step
        det = Detection(x1=x, y1=10.0, x2=x + 8.0, y2=30.0, conf=0.9, cls_id=0)
        return DetectionResult(
            detections=(det,), inference_time_s=0.0,
            model_name="fake", image_shape=(h, w),
        )


def _frame(i: int, h: int = 48, w: int = 64) -> np.ndarray:
    rng = np.random.RandomState(i)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_analyzer_emits_per_frame_signals_and_stride_windows():
    an = FrameAnalyzer(window=4, window_stride=2, grid_size=2, max_dim=None,
                       detector=FakeDetector())
    outs = [an.add_frame(_frame(i)) for i in range(6)]

    assert [o["frame_index"] for o in outs] == [0, 1, 2, 3, 4, 5]
    assert [o["count"] for o in outs] == [1] * 6
    # first frame has no previous frame -> zero flow
    assert outs[0]["flow_mean"] == 0.0

    windowed = [(i, o["window"]) for i, o in enumerate(outs) if o["window"]]
    assert [i for i, _ in windowed] == [3, 5]  # windows complete at len 4 and 6
    row0 = windowed[0][1]
    assert row0["window_start"] == 0 and row0["window_end"] == 4
    assert row0["person_count"] == 1.0
    assert isinstance(row0["mean_speed_pxf"], float)
    assert "directional_consistency" in row0
    assert windowed[1][1]["window_start"] == 2
    assert windowed[1][1]["window_end"] == 6


def test_analyzer_latest_window_rolling_and_none_until_full():
    an = FrameAnalyzer(window=4, window_stride=2, max_dim=None,
                       detector=FakeDetector())
    assert an.latest_window_features() is None
    for i in range(3):
        an.add_frame(_frame(i))
    assert an.latest_window_features() is None
    an.add_frame(_frame(3))
    assert an.latest_window_features()["window_start"] == 0
    an.add_frame(_frame(4))
    latest = an.latest_window_features()
    assert latest["window_start"] == 1  # rolling: last 4 of 5 frames
    assert latest["window_end"] == 5


def test_analyzer_reset_clears_state():
    an = FrameAnalyzer(window=4, window_stride=1, max_dim=None,
                       detector=FakeDetector())
    for i in range(5):
        an.add_frame(_frame(i))
    assert an.processed_frames == 5
    an.reset()
    assert an.processed_frames == 0
    assert an.latest_window_features() is None
    out = an.add_frame(_frame(0))
    assert out["frame_index"] == 0


def test_analyzer_respects_max_dim():
    an = FrameAnalyzer(window=2, window_stride=1, max_dim=48,
                       detector=FakeDetector())
    out = an.add_frame(_frame(0, h=96, w=128))
    assert max(out["shape"]) == 48
    assert out["shape"][0] < out["shape"][1]  # aspect preserved
