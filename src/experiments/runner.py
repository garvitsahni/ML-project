"""Experiments A-D on real feature tables; E stays gated without labels.

Honesty rules enforced here:
- grouped split by sequence_id (asserted disjoint; no frame leakage);
- metrics computed on the held-out TEST split only for reporting;
- Experiment E writes not_supported unless labeled congestion data exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.metrics import compute_binary_metrics
from src.features.schema import FeatureSchema
from src.models.splitting import assert_group_disjoint, grouped_split
from src.models.trainers import ALGORITHMS, feature_importance, train_model
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc,
)


@dataclass(frozen=True)
class ExperimentConfig:
    results_dir: Path
    figures_dir: Path
    seed: int = 42
    algorithms: tuple[str, ...] = ALGORITHMS
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class ExperimentResult:
    per_condition: dict = field(default_factory=dict)  # ablation: cond -> algo -> split -> metrics
    per_algorithm: dict = field(default_factory=dict)  # comparison: algo -> split -> metrics
    train_sequence_ids: tuple = ()
    test_sequence_ids: tuple = ()
    n_train_seqs: int = 0
    n_test_seqs: int = 0
    notes: list = field(default_factory=list)


META = {"sequence_id", "part", "split", "window_start", "window_end", "label"}


def load_feature_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature table not found: {path}. Run prepare_data.py first.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _schema_columns(df: pd.DataFrame) -> list[str]:
    schema = FeatureSchema.default()
    missing = [n for n in schema.names if n not in df.columns]
    if missing:
        raise ValueError(f"Feature table missing schema columns: {missing}")
    return list(schema.names)


def _split_frame(df: pd.DataFrame, config: ExperimentConfig):
    split = grouped_split(
        df["sequence_id"].to_numpy(),
        val_frac=config.val_frac, test_frac=config.test_frac, seed=config.seed,
    )
    assert_group_disjoint(df["sequence_id"].to_numpy(), split)
    return split


def _fit_score(Xtr, ytr, Xva, yva, Xte, yte, algo: str, seed: int,
               prefix: str, config: ExperimentConfig, tag: str) -> dict:
    model = train_model(algo, Xtr, ytr, seed=seed)
    out = {}
    for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        if len(X) == 0:
            out[name] = {"skipped": "empty split"}
            continue
        pred = model.predict(X)
        score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        out[name] = compute_binary_metrics(y, pred, score)
    # figures on the test split
    if len(Xte):
        pred = model.predict(Xte)
        cm = _confusion(yte, pred)
        plot_confusion_matrix(cm, ["normal", "anomalous"],
                              config.figures_dir / f"{prefix}_{tag}_confusion.png",
                              title=f"{tag} confusion")
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = _roc_curve_safe(yte, score)
            if fpr is not None:
                plot_roc(fpr, tpr, out["test"].get("roc_auc") or 0.0,
                         config.figures_dir / f"{prefix}_{tag}_roc.png")
        imp = feature_importance(model, list(Xte.columns))
        if imp is not None:
            plot_feature_importance(list(Xte.columns), imp,
                                    config.figures_dir / f"{prefix}_{tag}_importance.png")
    return out


def _confusion(y_true, y_pred) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _roc_curve_safe(y_true, y_score):
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return None, None, None
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr


def ablation_experiment(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """A static-only vs B movement-only vs C static+movement (schema groups)."""
    schema = FeatureSchema.default()
    static_cols = [s.name for s in schema.specs if s.group == "static"]
    movement_cols = [s.name for s in schema.specs if s.group == "movement"]
    split = _split_frame(df, config)
    y = df["label"].to_numpy()
    conditions = {
        "A_static_only": static_cols,
        "B_movement_only": movement_cols,
        "C_static_plus_movement": static_cols + movement_cols,
    }
    result = ExperimentResult(
        train_sequence_ids=tuple(sorted(set(df["sequence_id"].iloc[split.train]))),
        test_sequence_ids=tuple(sorted(set(df["sequence_id"].iloc[split.test]))),
    )
    for cond, cols in conditions.items():
        X = df[cols]
        result.per_condition[cond] = {}
        for algo in config.algorithms:
            metrics = _fit_score(
                X.iloc[split.train], y[split.train],
                X.iloc[split.val], y[split.val],
                X.iloc[split.test], y[split.test],
                algo, config.seed, f"ablation", config, f"{cond}_{algo}",
            )
            result.per_condition[cond][algo] = metrics
    result.n_train_seqs = len(result.train_sequence_ids)
    result.n_test_seqs = len(result.test_sequence_ids)
    _write_json(config.results_dir / "experiment_ablation.json",
                {"per_condition": result.per_condition,
                 "seed": config.seed,
                 "train_sequences": list(result.train_sequence_ids),
                 "test_sequences": list(result.test_sequence_ids)})
    return result


def algorithm_comparison(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """D: all five algorithms on static+movement features."""
    cols = _schema_columns(df)
    split = _split_frame(df, config)
    y = df["label"].to_numpy()
    X = df[cols]
    result = ExperimentResult(
        train_sequence_ids=tuple(sorted(set(df["sequence_id"].iloc[split.train]))),
        test_sequence_ids=tuple(sorted(set(df["sequence_id"].iloc[split.test]))),
    )
    for algo in config.algorithms:
        result.per_algorithm[algo] = _fit_score(
            X.iloc[split.train], y[split.train],
            X.iloc[split.val], y[split.val],
            X.iloc[split.test], y[split.test],
            algo, config.seed, "algorithms", config, algo,
        )
    result.n_train_seqs = len(result.train_sequence_ids)
    result.n_test_seqs = len(result.test_sequence_ids)
    _write_json(config.results_dir / "experiment_algorithms.json",
                {"per_algorithm": result.per_algorithm,
                 "seed": config.seed,
                 "train_sequences": list(result.train_sequence_ids),
                 "test_sequences": list(result.test_sequence_ids)})
    return result


def run_experiment_suite(
    table: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    seed: int = 42,
    algorithms: tuple[str, ...] = ALGORITHMS,
) -> dict:
    """Run A-D, gate E; returns a summary dict and writes status files."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(results_dir=results_dir, figures_dir=figures_dir,
                              seed=seed, algorithms=algorithms)
    abl = ablation_experiment(table, config)
    cmp_ = algorithm_comparison(table, config)
    # E: gated — congestion labels do not exist in this project
    (results_dir / "exp_e_status.json").write_text(
        json.dumps({
            "status": "not_supported",
            "reason": "No ground-truth congestion labels exist for this dataset; "
                      "Experiment E is intentionally disabled.",
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "complete",
        "experiments": {"ablation": True, "algorithms": True, "E": "not_supported"},
        "n_train_seqs": abl.n_train_seqs,
        "n_test_seqs": abl.n_test_seqs,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
