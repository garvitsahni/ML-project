"""Tests for experiment runner: ablation A-C, algorithm comparison D, gated E."""

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.metrics import compute_binary_metrics
from src.experiments.runner import (
    ExperimentConfig,
    ablation_experiment,
    algorithm_comparison,
    load_feature_table,
    run_experiment_suite,
)
from src.features.schema import FeatureSchema


def _synthetic_table(n_seq: int = 8, windows_per_seq: int = 6, seed: int = 0) -> pd.DataFrame:
    """Balanced synthetic table: last sequences are anomalous, separable on mean_count."""
    rng = np.random.default_rng(seed)
    schema = FeatureSchema.default()
    rows = []
    for s in range(n_seq):
        label_seq = s >= n_seq // 2
        for w in range(windows_per_seq):
            feats = {name: float(rng.normal()) for name in schema.names}
            feats["person_count"] = 25.0 if label_seq else 2.0
            feats["count_std"] = 6.0 if label_seq else 0.3
            rows.append({
                "sequence_id": f"seq{s:02d}",
                "part": "ped1",
                "split": "test" if s >= n_seq // 2 else "train",
                "window_start": w * 5,
                "window_end": w * 5 + 30,
                "label": int(label_seq),
                **feats,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def table():
    return _synthetic_table()


def test_compute_binary_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.9, 0.8])
    m = compute_binary_metrics(y_true, y_pred, y_score)
    assert set(m) >= {"accuracy", "precision", "recall", "f1", "roc_auc", "n_samples"}
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["n_samples"] == 4


def test_compute_metrics_single_class_auc_is_none():
    m = compute_binary_metrics(np.zeros(4, int), np.zeros(4, int), np.full(4, 0.3))
    assert m["roc_auc"] is None  # honest: undefined, not fabricated


def test_load_feature_table_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_table(tmp_path / "nope.parquet")


def test_ablation_experiment_returns_three_conditions(table, tmp_path):
    cfg = ExperimentConfig(results_dir=tmp_path, figures_dir=tmp_path / "figs", seed=42,
                           algorithms=("logreg", "dt"))
    result = ablation_experiment(table, config=cfg)
    assert set(result.per_condition) == {"A_static_only", "B_movement_only", "C_static_plus_movement"}
    for cond, algos in result.per_condition.items():
        for algo, splits in algos.items():
            assert "f1" in splits["test"] and "roc_auc" in splits["test"]


def test_ablation_combined_beats_or_equals_static_on_separable_data(table, tmp_path):
    cfg = ExperimentConfig(results_dir=tmp_path, figures_dir=tmp_path / "figs", seed=42,
                           algorithms=("logreg",))
    result = ablation_experiment(table, config=cfg)
    static_f1 = result.per_condition["A_static_only"]["logreg"]["test"]["f1"]
    combined_f1 = result.per_condition["C_static_plus_movement"]["logreg"]["test"]["f1"]
    assert combined_f1 >= static_f1 - 1e-9


def test_ablation_writes_json_and_figures(table, tmp_path):
    figs = tmp_path / "figs"
    cfg = ExperimentConfig(results_dir=tmp_path, figures_dir=figs, seed=42,
                           algorithms=("logreg", "dt"))
    result = ablation_experiment(table, config=cfg)
    out = tmp_path / "experiment_ablation.json"
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "per_condition" in payload
    assert list(figs.glob("*.png")), "expected at least one figure"


def test_algorithm_comparison_covers_all(table, tmp_path):
    cfg = ExperimentConfig(results_dir=tmp_path, figures_dir=tmp_path / "f", seed=42)
    result = algorithm_comparison(table, config=cfg)
    assert set(result.per_algorithm) == {"logreg", "dt", "rf", "svm", "xgb"}
    for algo, metrics in result.per_algorithm.items():
        assert "test" in metrics and "f1" in metrics["test"]


def test_group_split_respected(table, tmp_path):
    cfg = ExperimentConfig(results_dir=tmp_path, figures_dir=tmp_path / "f", seed=42,
                           algorithms=("logreg",))
    result = ablation_experiment(table, config=cfg)
    assert result.n_train_seqs > 0 and result.n_test_seqs > 0
    assert set(result.train_sequence_ids).isdisjoint(result.test_sequence_ids)


def test_run_experiment_suite_gates_e(tmp_path):
    out = run_experiment_suite(
        table=_synthetic_table(),
        results_dir=tmp_path,
        figures_dir=tmp_path / "figs",
        seed=42,
    )
    assert out["status"] == "complete"
    assert "ablation" in out["experiments"]
    assert "algorithms" in out["experiments"]
    e = json.loads((tmp_path / "exp_e_status.json").read_text(encoding="utf-8"))
    assert e["status"] == "not_supported"


def test_missing_labels_for_e_reports_reason(tmp_path):
    out = run_experiment_suite(
        table=_synthetic_table(),
        results_dir=tmp_path,
        figures_dir=tmp_path / "figs",
        seed=42,
    )
    status = json.loads((tmp_path / "exp_e_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "not_supported"
    assert "labels" in status["reason"].lower() or "ground" in status["reason"].lower()


# ------------------------------------------------------------- attribution
ATTRIBUTION_CONFIGS = {
    "baseline_old15_new_detector",
    "schema_v2_untuned",
    "tuned_threshold",
    "ensemble_final",
}


def _attr_table(n_seq: int = 10, windows_per_seq: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic v2 table with label_frac and a separable signal."""
    rng = np.random.default_rng(seed)
    schema = FeatureSchema.default()
    rows = []
    for s in range(n_seq):
        label_seq = s >= n_seq // 2
        for w in range(windows_per_seq):
            feats = {name: float(rng.normal()) for name in schema.names}
            feats["person_count"] = 25.0 if label_seq else 2.0
            rows.append({
                "sequence_id": f"seq{s:02d}", "part": "ped1",
                "split": "test" if label_seq else "train",
                "window_start": w * 5, "window_end": w * 5 + 30,
                "label": int(label_seq),
                "label_frac": 1.0 if label_seq else 0.0,
                **feats,
            })
    return pd.DataFrame(rows)


def test_run_attribution_four_configs_same_split(tmp_path):
    from src.experiments.attribution import run_attribution

    result = run_attribution(_attr_table(), seed=42, results_dir=tmp_path)
    assert set(result) == ATTRIBUTION_CONFIGS
    # identical split membership across configs
    splits = {cfg: tuple(payload["test_sequence_ids"]) for cfg, payload in result.items()}
    assert len(set(splits.values())) == 1


def test_run_attribution_payload_contents(tmp_path):
    from src.experiments.attribution import run_attribution

    result = run_attribution(_attr_table(), seed=42, results_dir=tmp_path)
    for cfg, payload in result.items():
        assert 0.0 < payload["threshold"] <= 1.0
        assert "test_metrics" in payload and "val_metrics" in payload
        meta = {"label", "label_frac", "sequence_id", "part", "split",
                "window_start", "window_end"}
        assert not (set(payload["columns"]) & meta), f"{cfg} uses meta columns"
    assert len(result["ensemble_final"]["members"]) == 3
    out = tmp_path / "experiment_attribution.json"
    assert out.is_file()  # provenance written


def test_run_attribution_baseline_uses_original_15(tmp_path):
    from src.experiments.attribution import ORIGINAL_15, run_attribution

    assert len(ORIGINAL_15) == 15
    result = run_attribution(_attr_table(), seed=42, results_dir=tmp_path)
    assert result["baseline_old15_new_detector"]["columns"] == ORIGINAL_15
    assert len(result["schema_v2_untuned"]["columns"]) > 15
