"""Configuration loading for CrowdSense AI.

All YAML configs live in ``configs/``. Relative paths in ``paths.yaml`` are
resolved against the project root so the project works regardless of the
invocation directory (Windows-safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when a config file is missing or cannot be parsed."""


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    metadata_dir: Path
    models_dir: Path
    results_dir: Path
    figures_dir: Path
    ucsd_dir: Path


def _read_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}. "
            f"Expected a YAML file; run from the project root or check configs/."
        )
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config {config_path} must contain a YAML mapping at top level.")
    return data


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_paths(config_path: Path | None = None) -> PathsConfig:
    """Load ``configs/paths.yaml`` with all paths resolved to absolute Paths."""
    path = config_path if config_path is not None else CONFIG_DIR / "paths.yaml"
    data = _read_yaml(Path(path))

    datasets = data.get("datasets") or {}
    ucsd = datasets.get("ucsd_anomaly") or {}
    if "path" not in ucsd:
        raise ConfigError(f"Config {path} is missing datasets.ucsd_anomaly.path")

    return PathsConfig(
        project_root=PROJECT_ROOT,
        raw_dir=_resolve(data["raw_root"]),
        processed_dir=_resolve(data["processed_root"]),
        metadata_dir=_resolve(data["metadata_root"]),
        models_dir=_resolve(data["models_dir"]),
        results_dir=_resolve(data["results_dir"]),
        figures_dir=_resolve(data["figures_dir"]),
        ucsd_dir=_resolve(ucsd["path"]),
    )


def load_experiment_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load ``configs/experiments.yaml`` as a nested dict."""
    path = config_path if config_path is not None else CONFIG_DIR / "experiments.yaml"
    return _read_yaml(Path(path))
