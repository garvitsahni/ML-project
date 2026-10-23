"""Shared dashboard helpers (pure where possible; safe to unit-test)."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_paths

# Config-driven directories (configs/paths.yaml is the single source of truth;
# MODELS_DIR previously pointed at a nonexistent reports/models, which made the
# model badge always read "not trained" despite five trained bundles).
MODELS_DIR = load_paths().models_dir
RESULTS_DIR = load_paths().results_dir
FIGURES_DIR = load_paths().figures_dir


def require_file(path: Path | str, hint: str = "") -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. {hint}".strip())
    return path


def list_result_files(directory: Path | str, suffix: str = ".json") -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == suffix)


def find_model_path(models_dir: Path | str = MODELS_DIR) -> Path | None:
    """Newest bundle in the models directory, or None."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return None
    bundles = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bundles[0] if bundles else None


def list_model_files(models_dir: Path | str = MODELS_DIR) -> list[tuple[Path, float]]:
    """All bundles with mtimes (newest first); [] when the directory is absent."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    bundles = [p for p in models_dir.iterdir()
               if p.is_file() and p.suffix == ".joblib"]
    return sorted(((p, p.stat().st_mtime) for p in bundles),
                  key=lambda item: item[1], reverse=True)


def load_evaluation(results_dir: Path | str = RESULTS_DIR) -> dict:
    """Parsed evaluation.json, or {} when missing/corrupt (never fabricates)."""
    path = Path(results_dir) / "evaluation.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def dataset_probe(root: Path | str, factory=None) -> dict:
    """Dataset availability and counts without loading frames.

    Returns {available, sequences, frames, anomalous, error}. Uses the real
    UcsdAdapter unless a factory is injected (tests). Counts reflect only
    what exists on disk.
    """
    root = Path(root)  # adapters expect Path; callers may pass str (cache keys)
    if factory is None:
        from src.datasets.ucsd_anomaly import UcsdAdapter

        factory = UcsdAdapter
    try:
        adapter = factory(root=root)
        seqs = list(adapter.iter_sequences())
        frames = sum(int(s.n_frames) for s in seqs)
        anomalous = sum(int(s.frame_labels.sum()) for s in seqs
                        if s.frame_labels is not None)
        return {"available": True, "sequences": len(seqs), "frames": frames,
                "anomalous": anomalous, "error": None}
    except Exception as exc:  # MissingDatasetError or any adapter failure
        return {"available": False, "sequences": None, "frames": None,
                "anomalous": None, "error": str(exc)}


def load_model_algorithm(model_path: Path | None) -> str | None:
    if model_path is None:
        return None
    try:
        from src.models.bundle import load_bundle

        return load_bundle(model_path).algorithm
    except Exception:
        return None


def model_status_badge(model_path: Path | None) -> tuple[str, str]:
    """(text, emoji) for sidebar status; pure so tests need no st runtime."""
    algo = load_model_algorithm(model_path)
    if algo:
        return f"model: {algo}", "✅"
    return "model: not trained", "⚠️"


def limitations() -> list[str]:
    """Real, current project limitations — shown on Methodology + Overview."""
    return [
        "Detection counting baseline only (YOLOv8n, CPU): no density CNN — no CUDA on this machine.",
        "No counting ground truth (QNRF/ShanghaiTech were dropped from scope): no MAE/RMSE reported.",
        "Speeds are px/frame in processed resolution — never m/s (no scene scale calibration).",
        "Window anomaly label = ANY anomalous frame in the window (UCSD frame-level GT).",
        "Grouped splits by sequence id: frames from one sequence never span train/val/test.",
        "Experiment E (congestion classification) is gated: no credible congestion labels exist.",
        "Dashboard renders only files that exist under reports/ — nothing is fabricated.",
    ]


def ucSD_placement_hint() -> str:
    return (
        "Place the UCSD Anomaly Dataset v1p2 under:\n"
        "  data/raw/ucsd_anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped1/{Train,Test}...\n"
        "Datasets are never auto-downloaded."
    )
