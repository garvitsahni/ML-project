"""Model bundles: joblib persistence with version + feature-schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

MODEL_VERSION = "1"


class ModelBundleError(RuntimeError):
    """Raised when a saved bundle is missing, corrupt, or incompatible."""


@dataclass
class ModelBundle:
    model: object  # fitted sklearn Pipeline
    feature_names: list[str]
    algorithm: str
    trained_at: str
    n_train: int
    metrics: dict
    threshold: float = 0.5  # decision threshold on P(anomaly)
    # sha256 of the full schema the trainer validated against (None for
    # legacy bundles saved before fingerprints -> exact-schema rule)
    schema_fingerprint: str | None = None


def save_bundle(bundle: ModelBundle, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": MODEL_VERSION,
            "model": bundle.model,
            "feature_names": list(bundle.feature_names),
            "algorithm": bundle.algorithm,
            "trained_at": bundle.trained_at,
            "n_train": bundle.n_train,
            "metrics": dict(bundle.metrics),
            "threshold": float(bundle.threshold),
            "schema_fingerprint": bundle.schema_fingerprint,
        },
        path,
    )
    return path


def load_bundle(path: Path | str) -> ModelBundle:
    path = Path(path)
    if not path.is_file():
        raise ModelBundleError(f"Model bundle not found: {path}")
    try:
        data = joblib.load(path)
    except Exception as exc:  # corrupt joblib
        raise ModelBundleError(f"Could not load bundle {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != MODEL_VERSION:
        raise ModelBundleError(
            f"Bundle {path} has version {data.get('version') if isinstance(data, dict) else '?'}; "
            f"expected {MODEL_VERSION} (retrain with train_ml.py)."
        )
    names = data.get("feature_names") or []
    if not names:
        raise ModelBundleError(f"Bundle {path} has no feature_names; retrain required.")
    return ModelBundle(
        model=data["model"],
        feature_names=list(names),
        algorithm=str(data.get("algorithm", "?")),
        trained_at=str(data.get("trained_at", "")),
        n_train=int(data.get("n_train", 0)),
        metrics=dict(data.get("metrics") or {}),
        threshold=float(data.get("threshold", 0.5)),
        schema_fingerprint=data.get("schema_fingerprint"),
    )


def predict(bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Predict labels; rejects feature-width mismatches (schema guard)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(bundle.feature_names):
        raise ValueError(
            f"Feature width mismatch: got {X.shape[1]}, bundle expects "
            f"{len(bundle.feature_names)} ({bundle.feature_names[:3]}...)"
        )
    return bundle.model.predict(X)


def predict_proba(bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(bundle.feature_names):
        raise ValueError("Feature width mismatch")
    return bundle.model.predict_proba(X)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
