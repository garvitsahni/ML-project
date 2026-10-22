"""Grouped-CV hyperparameter tuning, F1-optimal threshold, soft-vote helpers.

Honesty rules: GroupKFold groups = sequence_id (never random frames);
grids are pre-declared (no adaptive search); n_jobs=1 (Windows loky
pools crash intermittently).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.models.trainers import build_model

# Pre-declared small grids (spec Task 5) — bounded for CPU-only runtime.
GRIDS: dict[str, dict] = {
    "logreg": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "rf": {"clf__n_estimators": [200, 500], "clf__max_depth": [4, 8, None]},
    "xgb": {"clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.05, 0.1],
            "clf__n_estimators": [200, 400]},
    "svm": {"clf__C": [0.1, 1.0, 10.0]},
    "dt": {"clf__max_depth": [3, 5, None]},
}

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


def tune_model(
    name: str, X, y, groups: np.ndarray, seed: int = 42
) -> tuple[object, dict, float]:
    """Grouped-CV grid search on train data; returns (best_pipeline, params, cv_auc).

    ``groups`` are sequence ids — frames from one sequence stay in one fold.
    """
    if name not in GRIDS:
        raise ValueError(f"No tuning grid for '{name}'. Available: {sorted(GRIDS)}")
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError("groups length must match X")
    n_groups = len(np.unique(groups))
    search = GridSearchCV(
        build_model(name, seed=seed),
        GRIDS[name],
        cv=GroupKFold(n_splits=min(4, n_groups)),
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """F1-maximizing decision threshold on the pre-declared grid.

    Single-class y_true -> 0.5 (nothing to optimize); empty -> ValueError.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    if len(y_true) != len(scores):
        raise ValueError("y_true/scores length mismatch")
    if len(y_true) == 0:
        raise ValueError("No samples to threshold")
    if len(np.unique(y_true)) < 2:
        return 0.5
    f1s = [
        f1_score(y_true, scores >= t, zero_division=0) for t in THRESHOLD_GRID
    ]
    return float(THRESHOLD_GRID[int(np.argmax(f1s))])


def soft_vote(score_matrix: np.ndarray) -> np.ndarray:
    """Row-wise mean of per-model probability columns -> ensemble score."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(
            "score_matrix must be 2-D (n_samples, n_models) with >=1 model"
        )
    return matrix.mean(axis=1)
