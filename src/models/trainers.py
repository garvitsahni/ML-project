"""Five sklearn/xgboost pipelines; scalers fit inside the pipeline (train only)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str, seed: int = 42) -> Pipeline:
    """Return an UNFITTED pipeline (StandardScaler + classifier)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "dt":
        clf = DecisionTreeClassifier(random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=seed, eval_metric="logloss", n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


ALGORITHMS = ("logreg", "dt", "rf", "svm", "xgb")


def train_model(name: str, X: np.ndarray, y: np.ndarray, seed: int = 42) -> Pipeline:
    """Fit one pipeline on train data; returns the fitted pipeline."""
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if len(X) == 0:
        raise ValueError("Cannot train on zero samples")
    model = build_model(name, seed=seed)
    model.fit(X, y)
    return model


def train_all(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict[str, Pipeline]:
    """Fit every algorithm in ``ALGORITHMS`` on the same data."""
    return {name: train_model(name, X, y, seed=seed) for name in ALGORITHMS}


def feature_importance(model: Pipeline, feature_names: list[str]) -> np.ndarray | None:
    """Importance vector aligned to ``feature_names``, or None if unsupported."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_)).ravel()[: len(feature_names)]
    return None
