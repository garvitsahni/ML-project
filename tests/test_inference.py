"""Tests for the inference service: schema-strict window prediction."""

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FeatureSchema
from src.inference.service import InferenceError, InferenceService
from src.models.bundle import ModelBundle, save_bundle
from src.models.trainers import train_model


@pytest.fixture
def service(tmp_path):
    schema = FeatureSchema.default()
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, len(schema.names)))
    X[:, schema.names.index("person_count")] = np.r_[np.full(n // 2, 25.0), np.full(n - n // 2, 2.0)]
    y = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
    bundle = ModelBundle(
        model=train_model("logreg", X, y, seed=42),
        feature_names=list(schema.names),
        algorithm="logreg",
        trained_at="2026-10-18T00:00:00",
        n_train=n,
        metrics={"f1": 0.99},
    )
    path = save_bundle(bundle, tmp_path / "model.joblib")
    return InferenceService(path)


def _row(schema=None):
    schema = schema or FeatureSchema.default()
    return {name: 1.0 for name in schema.names} | {"person_count": 30.0}


def test_predict_row_returns_label_and_score(service):
    res = service.predict_row(_row())
    assert res.label in (0, 1)
    assert 0.0 <= res.score <= 1.0
    assert res.algorithm == "logreg"


def test_dense_row_scores_high_calm_row_scores_low(service):
    schema = FeatureSchema.default()
    dense = _row(schema)
    calm = {name: 1.0 for name in schema.names} | {"person_count": 1.0}
    assert service.predict_row(dense).score > service.predict_row(calm).score


def test_missing_feature_raises(service):
    row = _row()
    del row["mean_speed_pxf"]
    with pytest.raises(ValueError, match="mean_speed_pxf"):
        service.predict_row(row)


def test_extra_keys_ignored(service):
    row = _row() | {"bogus_column": 123}
    assert service.predict_row(row).label in (0, 1)


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(InferenceError, match="prepare|train|not found"):
        InferenceService(tmp_path / "none.joblib")


def test_row_order_follows_schema(service):
    shuffled = dict(reversed(list(_row().items())))
    res = service.predict_row(shuffled)
    assert res.label in (0, 1)  # reordering must not change anything
    res2 = service.predict_row(_row())
    assert res.score == pytest.approx(res2.score)


def _threshold_bundle_path(tmp_path, threshold):
    schema = FeatureSchema.default()
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, len(schema.names)))
    X[:, schema.names.index("person_count")] = np.r_[np.full(n // 2, 25.0), np.full(n - n // 2, 2.0)]
    y = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
    bundle = ModelBundle(
        model=train_model("logreg", X, y, seed=42),
        feature_names=list(schema.names),
        algorithm="logreg",
        trained_at="2026-10-18T00:00:00",
        n_train=n, metrics={}, threshold=threshold,
    )
    return save_bundle(bundle, tmp_path / f"m{threshold}.joblib")


def test_threshold_roundtrip_and_behavior(tmp_path):
    from src.models.bundle import load_bundle

    base = InferenceService(_threshold_bundle_path(tmp_path, 0.5))
    score = base.predict_row(_row()).score
    assert base.predict_row(_row()).label == 1  # default threshold keeps positive
    above = score + 0.05
    path = _threshold_bundle_path(tmp_path, above)
    assert load_bundle(path).threshold == above  # persistence round-trip
    # same row, threshold above its score -> downgraded to 0
    assert InferenceService(path).predict_row(_row()).label == 0


def test_bundle_without_threshold_key_defaults_to_half(tmp_path):
    import joblib

    from src.models.bundle import load_bundle

    schema = FeatureSchema.default()
    path = _threshold_bundle_path(tmp_path, 0.5)
    data = joblib.load(path)
    del data["threshold"]
    old = tmp_path / "old.joblib"
    joblib.dump(data, old)
    assert load_bundle(old).threshold == 0.5


def test_score_equal_to_threshold_is_positive(tmp_path):
    svc = InferenceService(_threshold_bundle_path(tmp_path, 0.5))
    row = _row()
    res = svc.predict_row(row)
    # craft threshold exactly at the score -> label must be 1 (>= semantics)
    svc.bundle.threshold = res.score
    assert svc.predict_row(row).label == 1
