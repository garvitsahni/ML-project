"""Presentation helpers shared by the API layer (pure; no UI framework).

These helpers were originally dashboard-only; the React + FastAPI frontend
reuses them so wording about model output stays identical across surfaces.
"""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT, load_paths

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


def find_model_path(models_dir: Path | str | None = None) -> Path | None:
    """Newest bundle in the configured models directory, or None."""
    if models_dir is None:
        try:
            models_dir = load_paths().models_dir
        except Exception:  # missing/invalid config must not crash status calls
            models_dir = MODELS_DIR
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return None
    bundles = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bundles[0] if bundles else None


def interpret_prediction(label: int, score: float) -> tuple[str, str]:
    """(headline, note) plain-language interpretation of one prediction.

    Scores are P(anomaly) from a modest sequence-held-out model — wording
    stays appropriately hedged.
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
