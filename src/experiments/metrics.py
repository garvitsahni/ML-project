"""Honest binary-classification metrics; undefined metrics stay None."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict:
    """Accuracy/precision/recall/f1 (+roc_auc when both classes present).

    ``roc_auc`` is None when a single class appears in ``y_true`` —
    never fabricated. All inputs are 1-D arrays of equal length.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true/y_pred length mismatch")
    if len(y_true) == 0:
        raise ValueError("No samples to score")

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_score).ravel()))
    else:
        out["roc_auc"] = None
    return out
