"""Shared dashboard helpers (pure where possible; safe to unit-test)."""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "reports" / "models"


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


def interpret_prediction(label: int, score: float) -> tuple[str, str]:
    """(headline, note) plain-language interpretation of one prediction.

    Pure so tests need no st runtime. Scores are P(anomaly) from a modest
    sequence-held-out model — wording stays appropriately hedged.
    """
    pct = f"{score * 100:.0f}%"
    if label == 1 and score >= 0.8:
        return "Anomaly likely", f"model confidence {pct}"
    if label == 1:
        return "Possible anomaly", f"low model confidence ({pct}) — treat as a hint"
    if score >= 0.3:
        return "Probably normal", f"score {pct} is near the decision boundary"
    return "Normal", f"model confidence {100 - score * 100:.0f}%"


def model_caveat() -> str:
    """One-line honesty caveat shown under every predictions block."""
    return (
        "Models are trained with sequence-held-out splits and are modest — "
        "see Methodology. Detector counts are estimates, never ground truth."
    )


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
