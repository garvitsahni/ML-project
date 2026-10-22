"""Tests for CLI scripts: help, graceful failure, tiny-fixture e2e."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures import make_ucsd_fixture

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(args, cwd, timeout=300):
    return subprocess.run(
        [sys.executable, *args], cwd=str(cwd),
        capture_output=True, text=True, timeout=timeout,
    )


def test_inspect_help(project_root):
    r = _run(["scripts/inspect_dataset.py", "--help"], project_root)
    assert r.returncode == 0


def test_prepare_detection_defaults_from_yaml():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_data", SCRIPTS / "prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    conf, imgsz = mod.resolve_detection_defaults(
        {"detection": {"conf": 0.10, "imgsz": 1280}})
    assert conf == 0.10 and imgsz == 1280
    with pytest.raises(KeyError):
        mod.resolve_detection_defaults({"detection": {"conf": 0.10}})


def test_inspect_graceful_when_root_absent(project_root, tmp_path):
    r = _run(["scripts/inspect_dataset.py", "--root", str(tmp_path / "empty")], project_root)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr + r.stdout
    assert "place" in (r.stdout + r.stderr).lower() or "not found" in (r.stdout + r.stderr).lower()


def test_prepare_help(project_root):
    r = _run(["scripts/prepare_data.py", "--help"], project_root)
    assert r.returncode == 0


def test_train_help(project_root):
    r = _run(["scripts/train_ml.py", "--help"], project_root)
    assert r.returncode == 0


def test_evaluate_help(project_root):
    r = _run(["scripts/evaluate.py", "--help"], project_root)
    assert r.returncode == 0


def test_evaluate_graceful_without_table(project_root, tmp_path):
    r = _run(["scripts/evaluate.py", "--table", str(tmp_path / "no.parquet")], project_root)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr + r.stdout


def test_full_cli_e2e_on_fixture(project_root, tmp_path):
    """prepare -> train (logreg only) -> evaluate on a synthetic UCSD fixture."""
    root = make_ucsd_fixture(tmp_path / "UCSD")
    table = tmp_path / "windows.parquet"
    out_dir = tmp_path / "out"
    r1 = _run([
        "scripts/prepare_data.py", "--root", str(root), "--window", "4", "--stride", "4",
        "--grid", "2", "--out", str(table),
    ], project_root, timeout=600)
    assert r1.returncode == 0, r1.stderr + r1.stdout
    assert table.is_file()
    meta = json.loads(table.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["rows"] > 0 and meta["sequences"] >= 1

    r2 = _run([
        "scripts/train_ml.py", "--table", str(table), "--algorithms", "logreg,dt",
        "--results-dir", str(out_dir / "results"), "--figures-dir", str(out_dir / "figures"),
        "--models-dir", str(out_dir / "models"),
    ], project_root, timeout=600)
    assert r2.returncode == 0, r2.stderr + r2.stdout
    assert (out_dir / "results" / "experiment_ablation.json").is_file()
    assert list((out_dir / "models").glob("*.joblib"))

    r3 = _run([
        "scripts/evaluate.py", "--table", str(table),
        "--results-dir", str(out_dir / "results"),
        "--models-dir", str(out_dir / "models"),
    ], project_root, timeout=600)
    assert r3.returncode == 0, r3.stderr + r3.stdout
    ev = json.loads((out_dir / "results" / "evaluation.json").read_text(encoding="utf-8"))
    assert "bundles" in ev and "logreg" in ev["bundles"]
    e_status = json.loads((out_dir / "results" / "exp_e_status.json").read_text(encoding="utf-8"))
    assert e_status["status"] == "not_supported"
