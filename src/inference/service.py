"""InferenceService: loads one trained bundle and scores feature rows.

Schema-strict with subset support: each bundle records a fingerprint of the
full schema it was trained against, so a bundle trained with screened-out
(`--exclude-features`) features is accepted while an older-schema bundle is
rejected with a retrain message. Legacy bundles saved before fingerprints
existed must match the schema exactly. Rows are keyed to the bundle's own
feature order; extra keys are ignored so upstream tables may carry metadata
columns.
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
        fp = self.bundle.schema_fingerprint
        if fp is None:
            # legacy bundle (pre-fingerprint): require the exact full schema,
            # otherwise the plain subset relation would accept a v1 bundle
            if set(self.bundle.feature_names) != set(self.schema.names):
                missing = [
                    n for n in self.schema.names
                    if n not in self.bundle.feature_names
                ]
                raise InferenceError(
                    f"Bundle at {model_path} predates schema features "
                    f"{missing}; retrain."
                )
        elif fp != self.schema.fingerprint:
            raise InferenceError(
                f"Bundle at {model_path} was trained under a different "
                "feature schema; retrain (python scripts/train_ml.py)."
            )
        else:
            # fingerprint matches: the bundle may use a screened subset of
            # schema features (spec design: select the bundle's subset)
            unknown = [
                n for n in self.bundle.feature_names
                if n not in self.schema.names
            ]
            if unknown:
                raise InferenceError(
                    f"Bundle at {model_path} uses features outside the "
                    f"current schema: {unknown}; retrain."
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
        """Score one window's feature row -> label + anomaly score.

        Label = 1 iff P(anomaly) >= bundle.threshold; models without
        predict_proba fall back to the estimator's own predict().
        """
        x = self._to_vector(row).reshape(1, -1)
        model = self.bundle.model
        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", [0, 1]))
            proba = model.predict_proba(x)[0]
            score = float(proba[classes.index(1)]) if 1 in classes else float(model.predict(x)[0])
            label = int(score >= self.bundle.threshold)
        else:
            label = int(model.predict(x)[0])
            score = float(label)
        return InferenceResult(
            label=label,
            score=score,
            algorithm=self.bundle.algorithm,
            model_trained_at=self.bundle.trained_at,
        )

    def predict_rows(self, rows: list[dict]) -> list[InferenceResult]:
        return [self.predict_row(r) for r in rows]
