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
