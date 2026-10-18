"""InferenceService: loads one trained bundle and scores feature rows.

Schema-strict: every schema feature must be present (order irrelevant —
rows are re-keyed to the bundle's feature order). Extra keys are ignored
so upstream tables may carry metadata columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.schema import FeatureSchema
from src.models.bundle import ModelBundle, ModelBundleError, load_bundle


class InferenceError(RuntimeError):
    """No usable model (missing/corrupt bundle) — message says how to fix."""


@dataclass(frozen=True)
class InferenceResult:
    label: int
    score: float  # P(anomaly) from predict_proba, or label as fallback
    algorithm: str
    model_trained_at: str


class InferenceService:
    def __init__(self, model_path: Path | str):
        try:
            self.bundle: ModelBundle = load_bundle(model_path)
        except ModelBundleError as exc:
            raise InferenceError(
                f"{exc} Train a model first (python scripts/train_ml.py) or "
                "check configs/paths.yaml."
            ) from exc
        self.schema = FeatureSchema.default()
        missing = [n for n in self.schema.names if n not in self.bundle.feature_names]
        if missing:
            raise InferenceError(
                f"Bundle at {model_path} predates schema features {missing}; retrain."
            )

    def _to_vector(self, row: dict) -> np.ndarray:
        missing = [n for n in self.bundle.feature_names if n not in row]
        if missing:
            raise ValueError(
                f"Feature row missing required features: {missing} "
                f"(expected all of {self.bundle.feature_names})"
            )
        return np.array([float(row[n]) for n in self.bundle.feature_names], dtype=float)

    def predict_row(self, row: dict) -> InferenceResult:
        """Score one window's feature row -> label + anomaly score."""
        x = self._to_vector(row).reshape(1, -1)
        model = self.bundle.model
        label = int(model.predict(x)[0])
        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", [0, 1]))
            proba = model.predict_proba(x)[0]
            score = float(proba[classes.index(1)]) if 1 in classes else float(label)
        else:
            score = float(label)
        return InferenceResult(
            label=label,
            score=score,
            algorithm=self.bundle.algorithm,
            model_trained_at=self.bundle.trained_at,
        )

    def predict_rows(self, rows: list[dict]) -> list[InferenceResult]:
        return [self.predict_row(r) for r in rows]
