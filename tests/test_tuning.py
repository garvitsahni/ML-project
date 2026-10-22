"""Tests for grouped-CV tuning, F1-optimal threshold, and soft-vote helpers."""

import numpy as np
import pytest

from src.models.tuning import GRIDS, best_threshold, soft_vote, tune_model


@pytest.fixture
def separable_groups():
    """8 groups x 10 samples; feature 0 separates classes group-independently."""
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(8), 10)
    y = np.tile([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 8)
    X = rng.normal(size=(80, 3))
    X[:, 0] = y * 3.0 + rng.normal(scale=0.1, size=80)
    return X, y, groups


def test_grids_cover_all_algorithms():
    from src.models.trainers import ALGORITHMS

    assert set(GRIDS) == set(ALGORITHMS)


def test_tune_model_returns_fitted_pipeline_and_params(separable_groups):
    X, y, groups = separable_groups
    model, params, cv_auc = tune_model("logreg", X, y, groups, seed=42)
    assert set(params) <= set(GRIDS["logreg"])
    assert 0.0 <= cv_auc <= 1.0
    assert hasattr(model, "predict_proba")
    # refit on all training data
    assert len(model.predict(X)) == len(X)


def test_tune_model_group_cv_holds_out_groups():
    """A group-only memorizer must not score perfectly under grouped CV."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(6), 10)
    # label = group parity: learnable per-group, not across groups
    y = np.tile([g % 2 for g in range(6)], 10).astype(int)
    X = rng.normal(size=(60, 5))
    X[:, 0] = groups % 2 + rng.normal(scale=0.05, size=60)  # leaky-ish feature
    _, _, cv_auc = tune_model("logreg", X, y, groups, seed=42)
    assert cv_auc < 1.0  # grouped CV withholds whole groups


def test_tune_model_unknown_algorithm_raises(separable_groups):
    X, y, groups = separable_groups
    with pytest.raises(ValueError):
        tune_model("nope", X, y, groups)


def test_best_threshold_maximizes_f1():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.45, 0.4, 0.7, 0.9])
    # threshold 0.05..0.45 buckets: at t<=0.4 -> preds 0,0,0? (0.45>=0.4 -> 1)
    # best F1 at t=0.45: preds 0,0,0,0,1,1 -> f1 = 2*1*1/(1+1+2*1)? compute via loop
    best = best_threshold(y_true, scores)
    from sklearn.metrics import f1_score

    f1s = [f1_score(y_true, scores >= t, zero_division=0)
           for t in np.arange(0.05, 0.96, 0.05)]
    assert best == pytest.approx(np.arange(0.05, 0.96, 0.05)[int(np.argmax(f1s))])


def test_best_threshold_degenerate_cases():
    assert best_threshold(np.array([1, 1]), np.array([0.9, 0.8])) == 0.5  # single class
    with pytest.raises(ValueError):
        best_threshold(np.array([]), np.array([]))


def test_soft_vote_averages_columns():
    matrix = np.array([[0.9, 0.1], [0.2, 0.4]])
    np.testing.assert_allclose(soft_vote(matrix), [0.5, 0.3])


def test_soft_vote_validizes_shape():
    with pytest.raises(ValueError):
        soft_vote(np.array([0.9, 0.1, 0.4]))  # 1-D: not a (samples, models) matrix
    with pytest.raises(ValueError):
        soft_vote(np.zeros((3, 0)))  # no models
