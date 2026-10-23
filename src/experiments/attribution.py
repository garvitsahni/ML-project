"""The four pre-declared test-split evaluations (spec Goal 3).

This module is the ONLY sanctioned code that evaluates on the held-out
TEST split beyond the existing A-D suite: exactly four configs, each
logged with full provenance (columns, params, threshold, split ids).
All selection decisions (tuning, threshold, ensemble membership) use
train-side GroupKFold CV and the VAL split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.metrics import compute_binary_metrics
from src.models.splitting import assert_group_disjoint, grouped_split
from src.models.trainers import train_model
from src.models.tuning import best_threshold, soft_vote, tune_model

# The pre-schema-v2 feature set (git history of src/features/schema.py).
ORIGINAL_15 = [
    "person_count", "density_per_mpx", "grid_concentration", "spread_x",
    "spread_y", "grid_entropy", "density_delta", "mean_speed_pxf",
    "speed_std_pxf", "direction_entropy", "directional_consistency",
    "flow_mean_magnitude", "flow_std_pxf", "track_count", "track_persistence",
]

ENSEMBLE_MEMBERS = 3


def _proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def _evaluate(model, X, y, threshold: float) -> tuple[dict, np.ndarray]:
    scores = _proba(model, X)
    metrics = compute_binary_metrics(y, (scores >= threshold).astype(int), scores)
    return metrics, scores


def run_attribution(
    df: pd.DataFrame, seed: int = 42, results_dir: Path | str = "reports/results"
) -> dict:
    """Run the four pre-declared configs; write experiment_attribution.json."""
    results_dir = Path(results_dir)
    split = grouped_split(df["sequence_id"].to_numpy(), seed=seed)
    assert_group_disjoint(df["sequence_id"].to_numpy(), split)
    y = df["label"].to_numpy()
    groups = df["sequence_id"].to_numpy()[split.train]
    test_seq_ids = sorted(set(df["sequence_id"].iloc[split.test]))
    val_seq_ids = sorted(set(df["sequence_id"].iloc[split.val]))

    def fit_eval(cols: list[str], model, threshold: float) -> dict:
        X = df[cols]
        val_metrics, _ = _evaluate(model, X.iloc[split.val], y[split.val], threshold)
        test_metrics, _ = _evaluate(model, X.iloc[split.test], y[split.test], threshold)
        return {
            "columns": list(cols),
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_sequence_ids": test_seq_ids,
        }

    all_features = [c for c in df.columns if c not in {
        "sequence_id", "part", "split", "window_start", "window_end",
        "label", "label_frac",
    }]
    X = df[all_features]

    out: dict[str, dict] = {}

    # 1. baseline: original 15 features, untuned, 0.5 threshold
    base_model = train_model("logreg", X[ORIGINAL_15].iloc[split.train],
                             y[split.train], seed=seed)
    out["baseline_old15_new_detector"] = fit_eval(ORIGINAL_15, base_model, 0.5)

    # 2. full schema v2, untuned, 0.5
    v2_model = train_model("logreg", X.iloc[split.train], y[split.train], seed=seed)
    out["schema_v2_untuned"] = fit_eval(all_features, v2_model, 0.5)

    # 3. per-algorithm tuning on train (grouped CV) + threshold on val
    tuned: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for algo in ("logreg", "rf", "xgb", "svm", "dt"):
        model, params, cv_auc = tune_model(
            algo, X.iloc[split.train], y[split.train], groups, seed=seed
        )
        fitted[algo] = model
        val_scores = _proba(model, X.iloc[split.val])
        threshold = best_threshold(y[split.val], val_scores)
        entry = fit_eval(all_features, model, threshold)
        entry["best_params"] = params
        entry["cv_auc"] = cv_auc
        tuned[algo] = entry

    best_algos = sorted(
        tuned,
        key=lambda a: (tuned[a]["val_metrics"].get("roc_auc") is None,
                       -(tuned[a]["val_metrics"].get("roc_auc") or 0.0)),
    )[:ENSEMBLE_MEMBERS]

    # the tuned_threshold entry reports the best single algorithm by val AUC;
    # per_algorithm carries every tuned entry (thresholds + provenance) for
    # train_ml.py --tune to persist bundles without touching test again
    best_single = best_algos[0]
    out["tuned_threshold"] = {
        "best_algorithm": best_single,
        "per_algorithm": tuned,
        **tuned[best_single],
    }

    # 4. soft-vote ensemble of the top-3 by val AUC; threshold on val
    ensemble_val_scores = soft_vote(np.column_stack([
        _proba(fitted[a], X.iloc[split.val]) for a in best_algos
    ]))
    ensemble_threshold = best_threshold(y[split.val], ensemble_val_scores)
    ensemble_test_scores = soft_vote(np.column_stack([
        _proba(fitted[a], X.iloc[split.test]) for a in best_algos
    ]))
    out["ensemble_final"] = {
        "members": best_algos,
        "columns": list(all_features),
        "threshold": float(ensemble_threshold),
        "val_metrics": compute_binary_metrics(
            y[split.val], (ensemble_val_scores >= ensemble_threshold).astype(int),
            ensemble_val_scores),
        "test_metrics": compute_binary_metrics(
            y[split.test], (ensemble_test_scores >= ensemble_threshold).astype(int),
            ensemble_test_scores),
        "test_sequence_ids": test_seq_ids,
    }

    payload = {
        "configs": out,
        "seed": seed,
        "train_sequence_ids": sorted(set(df["sequence_id"].iloc[split.train])),
        "val_sequence_ids": val_seq_ids,
        "test_sequence_ids": test_seq_ids,
        "note": "pre-declared four-config test evaluation (spec Goal 3); "
                "test touched exactly once per config",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_attribution.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return out
