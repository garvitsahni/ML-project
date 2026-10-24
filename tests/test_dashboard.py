"""Dashboard tests: imports, helper purity, headless-start smoke."""

import importlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.ui import common

PAGES = [
    "page_overview", "page_photo", "page_video",
    "page_camera", "page_settings",
]


def test_app_module_imports():
    mod = importlib.import_module("app")
    assert hasattr(mod, "main")


@pytest.mark.parametrize("name", PAGES)
def test_page_module_imports_with_render(name):
    mod = importlib.import_module(f"src.ui.{name}")
    assert callable(getattr(mod, "render"))


def test_format_bytes_units():
    from src.ui import common

    assert common.format_bytes(512) == "512 B"
    assert common.format_bytes(25 * 1024) == "25 KB"
    assert common.format_bytes(3 * 1024 * 1024) == "3.0 MB"


def test_require_file_returns_path(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    assert common.require_file(f, hint="run train") == f


def test_require_file_missing_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="run train"):
        common.require_file(tmp_path / "no.json", hint="run train")


def test_list_result_files_sorted_and_filtered(tmp_path):
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    out = common.list_result_files(tmp_path, suffix=".json")
    assert [p.name for p in out] == ["a.json", "b.json"]


def test_list_result_files_empty_dir(tmp_path):
    assert common.list_result_files(tmp_path, suffix=".json") == []


def test_model_status_badge_pure(tmp_path):
    text, emoji = common.model_status_badge(None)
    assert "not" in text.lower()
    # with a real bundle on disk the badge names the algorithm
    import numpy as np

    from src.models.bundle import ModelBundle, save_bundle
    from src.models.trainers import train_model

    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.array([0, 1] * 10)
    bundle = ModelBundle(model=train_model("logreg", X, y, seed=1),
                         feature_names=["a", "b", "c"], algorithm="logreg",
                         trained_at="2026-10-18", n_train=20, metrics={})
    path = save_bundle(bundle, tmp_path / "m.joblib")
    text2, emoji2 = common.model_status_badge(path)
    assert "logreg" in text2


def test_photo_page_decodes_upload_and_analyzes_in_memory():
    """Regression: the photo page must analyze the decoded ndarray in memory
    (analyze_frame) — it previously passed the array to analyze_image, which
    only accepts disk paths, so every upload crashed with a TypeError."""
    import numpy as np

    from src.detection.detector import Detection, DetectionResult
    from src.ui import page_photo

    class FakeDetector:
        def detect(self, frame):
            h, w = frame.shape[:2]
            return DetectionResult(
                detections=(Detection(x1=10, y1=10, x2=30, y2=40, conf=0.9, cls_id=0),),
                inference_time_s=0.0,
                model_name="fake",
                image_shape=(h, w),
            )

    img = np.zeros((64, 96, 3), dtype=np.uint8)
    ok, buf = __import__("cv2").imencode(".png", img)
    assert ok
    image, analysis = page_photo.analyze_upload(buf.tobytes(), detector=FakeDetector())
    assert analysis.count == 1
    assert image.shape[:2] == (64, 96)


def test_validate_upload_accepts_valid_files():
    from src.ui import page_photo

    assert page_photo.validate_upload("crowd.jpg", 1024, "image/jpeg") is None
    assert page_photo.validate_upload("a.PNG", 1024, None) is None
    assert page_photo.validate_upload("scan.tif", 5_000_000, "image/tiff") is None


def test_validate_upload_rejects_wrong_extension():
    from src.ui import page_photo

    err = page_photo.validate_upload("movie.mp4", 1024, "video/mp4")
    assert err is not None
    assert "mp4" in err


def test_validate_upload_rejects_oversized_file():
    from src.ui import page_photo

    err = page_photo.validate_upload("big.jpg", 21 * 1024 * 1024, "image/jpeg")
    assert err is not None
    assert "20 MB" in err


def test_validate_upload_rejects_empty_and_missing_file():
    from src.ui import page_photo

    assert page_photo.validate_upload("x.jpg", 0, "image/jpeg") is not None
    assert page_photo.validate_upload(None, 100, None) is not None


def test_validate_upload_rejects_non_image_content_type():
    from src.ui import page_photo

    err = page_photo.validate_upload("x.jpg", 100, "text/plain")
    assert err is not None
    assert "image" in err


def test_validate_video_upload_accepts_and_rejects():
    from src.ui import page_video

    assert page_video.validate_video_upload("clip.mp4", 1024) is None
    assert page_video.validate_video_upload("clip.mov", 1024) is None
    err = page_video.validate_video_upload("photo.jpg", 1024)
    assert err is not None and "mp4" in err
    err = page_video.validate_video_upload("big.mp4", 201 * 1024 * 1024)
    assert err is not None and "200 MB" in err
    assert page_video.validate_video_upload(None, 10) is not None
    assert page_video.validate_video_upload("empty.mp4", 0) is not None


def test_fmt_size_units():
    from src.ui import page_video

    assert page_video._fmt_size(25 * 1024) == "25 KB"
    assert page_video._fmt_size(3 * 1024 * 1024) == "3.0 MB"
    assert page_video._fmt_size(1024 * 1024) == "1.0 MB"


class _FakeCameraCap:
    """Minimal capture double: yields n synthetic frames, then EOF."""

    def __init__(self, n: int):
        self._n = n
        self._i = 0
        self.released = False

    def read(self):
        if self._i >= self._n:
            return False, None
        import numpy as np

        frame = np.random.RandomState(self._i).randint(
            0, 255, (48, 64, 3), dtype=np.uint8)
        self._i += 1
        return True, frame

    def release(self):
        self.released = True


class _FakeDetector:
    def detect(self, frame):
        from src.detection.detector import Detection, DetectionResult

        h, w = frame.shape[:2]
        return DetectionResult(
            detections=(Detection(x1=10, y1=10, x2=30, y2=40,
                                  conf=0.9, cls_id=0),),
            inference_time_s=0.0, model_name="fake", image_shape=(h, w))


def _camera_store(n_frames: int, stride: int = 1):
    from src.ui.analysis_loop import FrameAnalyzer
    from src.ui import page_camera

    return {
        "camera_cap": _FakeCameraCap(n_frames),
        "camera_analyzer": FrameAnalyzer(
            window=4, window_stride=2, max_dim=None, detector=_FakeDetector()),
        "camera_cfg": {"stride": stride, "max_dim": None},
    }


def test_step_capture_processes_injected_fake_cap():
    from src.ui import page_camera

    store = _camera_store(8)
    out = page_camera.step_capture(5, store=store)
    assert out["read"] == 5 and out["processed"] == 5
    assert out["stream_ended"] is False
    assert out["last"]["frame_index"] == 4
    assert out["fps"] > 0 and out["latency_ms"] >= 0
    assert out["frame"].shape == (48, 64, 3)

    # continues where the previous batch stopped; EOF after 8 frames
    out2 = page_camera.step_capture(4, store=store)
    assert out2["read"] == 4 and out2["processed"] == 3
    assert out2["stream_ended"] is True
    assert out2["last"]["frame_index"] == 7
    assert out2["window"] is not None
    assert out2["window"]["window_start"] == 4  # rolling: last 4 of 8 frames


def test_step_capture_honors_stride():
    from src.ui import page_camera

    store = _camera_store(10, stride=2)
    out = page_camera.step_capture(10, store=store)
    assert out["read"] == 10
    assert out["processed"] == 5  # frames 0, 2, 4, 6, 8


def test_overlay_grid_draws_cells_on_copy():
    import numpy as np

    from src.ui import page_camera

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    grid = np.array([[0, 5], [1, 0]])
    out = page_camera._overlay_grid(frame, grid)
    assert out.shape == frame.shape
    assert out.max() > 0                      # tinted cells changed pixels
    assert out[:20, :30].max() == 0           # zero-count cell untouched
    assert np.array_equal(frame, np.zeros_like(frame))  # original intact


def test_probe_devices_uses_injected_opener_and_caches():
    from src.ui import page_camera

    class FakeCap:
        def __init__(self, ok):
            self._ok = ok

        def isOpened(self):
            return self._ok

        def release(self):
            pass

    availability = {0: True, 1: False, 2: True}
    store: dict = {}
    devices = page_camera._probe_devices(
        2, opener=lambda i: FakeCap(availability[i]), store=store)
    assert devices == [0, 2]
    assert store["camera_devices"] == [0, 2]


def test_diagnostics_whitelisted_only():
    from src.ui import page_settings

    diag = page_settings.diagnostics()
    allowed = {"python", "streamlit", "torch_device", "dataset_present",
               "model_present", "results_present"}
    assert set(diag) <= allowed
    assert diag["python"] and diag["python"][0].isdigit()
    assert diag["streamlit"]
    assert isinstance(diag["dataset_present"], bool)
    assert isinstance(diag["model_present"], bool)
    assert isinstance(diag["results_present"], bool)
    for value in diag.values():
        text = str(value)
        assert ":\\" not in text            # no absolute paths
        assert "KMP_DUPLICATE" not in text  # no env values


def test_configured_detector_uses_settings_conf():
    from src.ui import common

    det = common.configured_detector(
        store={"settings": {"conf_threshold": 0.6}})
    assert det.conf == 0.6
    det = common.configured_detector(store={})
    assert det.conf == 0.25


def test_list_model_files_sorted_newest_first_and_filtered(tmp_path):
    import os

    old = tmp_path / "old.joblib"
    new = tmp_path / "new.joblib"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (1, 1))  # epoch -> definitively older
    (tmp_path / "skip.txt").write_text("z", encoding="utf-8")
    out = common.list_model_files(tmp_path)
    assert [p.name for p, _mtime in out] == ["new.joblib", "old.joblib"]
    assert isinstance(out[0][1], float)


def test_list_model_files_missing_dir_returns_empty(tmp_path):
    assert common.list_model_files(tmp_path / "nope") == []


def test_load_evaluation_missing_returns_empty(tmp_path):
    assert common.load_evaluation(tmp_path) == {}


def test_load_evaluation_reads_existing_json(tmp_path):
    (tmp_path / "evaluation.json").write_text(
        '{"logreg": {"test": {"roc_auc": 0.675}}}', encoding="utf-8")
    out = common.load_evaluation(tmp_path)
    assert out["logreg"]["test"]["roc_auc"] == 0.675


def test_load_evaluation_corrupt_file_returns_empty(tmp_path):
    (tmp_path / "evaluation.json").write_text("{not json", encoding="utf-8")
    assert common.load_evaluation(tmp_path) == {}


def test_dataset_probe_with_fake_adapter_counts():
    import numpy as np

    class Seq:
        def __init__(self, n, labels):
            self.n_frames = n
            self.frame_labels = labels

    class FakeAdapter:
        def __init__(self, root=None):
            pass

        def iter_sequences(self):
            return iter([Seq(10, np.array([0, 1] * 5)), Seq(5, np.zeros(5, dtype=int))])

    probe = common.dataset_probe(Path("any"), factory=FakeAdapter)
    assert probe == {"available": True, "sequences": 2, "frames": 15,
                     "anomalous": 5, "error": None}


def test_dataset_probe_missing_reports_unavailable(tmp_path):
    class Boom:
        def __init__(self, root=None):
            raise FileNotFoundError("dataset not found")

    probe = common.dataset_probe(tmp_path / "x", factory=Boom)
    assert probe["available"] is False
    assert probe["error"]
    assert probe["sequences"] is None


def test_dataset_probe_accepts_string_root():
    from pathlib import Path

    seen = {}

    class Fake:
        def __init__(self, root=None):
            seen["root"] = root

        def iter_sequences(self):
            return iter([])

    probe = common.dataset_probe("data/raw/some_root", factory=Fake)
    assert probe["available"] is True
    # the adapter expects Path objects: dataset_probe normalizes at its boundary
    assert isinstance(seen["root"], Path)


def test_streamlit_headless_smoke(project_root):
    """App must start headless with no traceback within ~8s."""
    port = 8599
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=str(project_root),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 8
        output = ""
        while time.time() < deadline:
            line = proc.stdout.readline()
            if line:
                output += line
                if "You can now view" in output or "Traceback" in output:
                    break
        assert "Traceback" not in output, output
        assert "You can now view" in output or "Traceback" not in output
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_status_items_reports_real_state():
    from src.ui.page_overview import status_items

    ok_probe = {"available": True, "sequences": 200}
    ev = {"bundles": {"xgb": {}}, "n_test_windows": 48}
    assert status_items(3, ok_probe, ev, 2) == [
        ("ok", "3 model bundles ready"),
        ("ok", "Dataset ready \u00b7 200 sequences"),
        ("idle", "48 evaluation windows"),
        ("idle", "2 runs this session"),
    ]
    assert status_items(0, {"available": False}, {}, 0) == [
        ("warn", "No models yet"),
        ("warn", "Dataset not found"),
        ("idle", "No evaluation yet"),
        ("idle", "No runs yet"),
    ]
    assert status_items(1, ok_probe, ev, 1)[0] == ("ok", "1 model bundle ready")


def test_photo_step_reflects_upload_and_result():
    from src.ui.page_photo import photo_step

    assert photo_step(False, False) == 1
    assert photo_step(True, False) == 2
    assert photo_step(True, True) == 3
    assert photo_step(False, True) == 3   # result wins over stale upload flag
