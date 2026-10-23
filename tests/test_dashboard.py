"""Dashboard tests: imports, helper purity, headless-start smoke."""

import importlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.ui import common

PAGES = [
    "page_overview", "page_methodology", "page_datasets", "page_photo",
    "page_video", "page_camera", "page_training", "page_results",
]


def test_app_module_imports():
    mod = importlib.import_module("app")
    assert hasattr(mod, "main")


@pytest.mark.parametrize("name", PAGES)
def test_page_module_imports_with_render(name):
    mod = importlib.import_module(f"src.ui.{name}")
    assert callable(getattr(mod, "render"))


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
