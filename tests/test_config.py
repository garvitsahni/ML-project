"""Tests for the config loader (src/config.py)."""

from pathlib import Path

import pytest

from src.config import ConfigError, PathsConfig, load_experiment_config, load_paths


def test_load_paths_reads_project_config():
    paths = load_paths()
    assert isinstance(paths, PathsConfig)
    assert paths.ucsd_dir.name == "UCSD_Anomaly_Dataset.v1p2"
    assert paths.ucsd_dir.is_absolute()
    assert str(paths.ucsd_dir).endswith(
        str(Path("data") / "raw" / "ucsd_anomaly" / "UCSD_Anomaly_Dataset.v1p2")
    )


def test_load_paths_resolves_all_roots_under_project():
    paths = load_paths()
    for attr in (
        "raw_dir",
        "processed_dir",
        "metadata_dir",
        "models_dir",
        "results_dir",
        "figures_dir",
    ):
        value = getattr(paths, attr)
        assert isinstance(value, Path), attr
        assert value.is_absolute(), attr


def test_load_paths_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_paths(config_path=tmp_path / "nope.yaml")
    assert "nope.yaml" in str(exc.value)


def test_load_paths_invalid_yaml_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("::: not yaml {{{", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_paths(config_path=bad)


def test_load_experiment_config_returns_nested_dict():
    cfg = load_experiment_config()
    assert cfg["experiments"]["algorithms"] == ["logreg", "dt", "rf", "svm", "xgb"]
    assert cfg["experiments"]["prediction_horizons"]["enabled"] is False
    assert cfg["seed"] == 42


def test_load_experiment_config_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_experiment_config(config_path=tmp_path / "missing.yaml")
