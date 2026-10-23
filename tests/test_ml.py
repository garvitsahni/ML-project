"""Tests for grouped splitting, model training, and joblib bundles."""

import numpy as np
import pytest

from src.models.bundle import (
    MODEL_VERSION,
    ModelBundle,
    ModelBundleError,
    load_bundle,
    predict,
    save_bundle,
)
from src.models.splitting import assert_group_disjoint, grouped_split
from src.models.trainers import ALGORITHMS, train_all, train_model


@pytest.fixture
def synthetic_xy():
    rng = np.random.default_rng(7)
    n, g = 120, 24
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    groups = np.repeat([f"g{i:02d}" for i in range(g)], n // g)
    return X, y, groups


# ---------------------------------------------------------------- splitting
def test_grouped_split_disjoint(synthetic_xy):
    _, _, groups = synthetic_xy
    split = grouped_split(groups, val_frac=0.2, test_frac=0.2, seed=42)
    g = lambda idx: set(groups[idx])
    assert g(split.train).isdisjoint(g(split.val))
    assert g(split.train).isdisjoint(g(split.test))
    assert g(split.val).isdisjoint(g(split.test))
    assert_group_disjoint(groups, split)


def test_grouped_split_covers_all_indices(synthetic_xy):
    _, _, groups = synthetic_xy
    split = grouped_split(groups, val_frac=0.2, test_frac=0.2, seed=42)
    combined = np.concatenate([split.train, split.val, split.test])
    assert sorted(combined.tolist()) == list(range(len(groups)))


def test_grouped_split_deterministic(synthetic_xy):
    _, _, groups = synthetic_xy
    a = grouped_split(groups, val_frac=0.2, test_frac=0.2, seed=42)
    b = grouped_split(groups, val_frac=0.2, test_frac=0.2, seed=42)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.test, b.test)


def test_assert_group_disjoint_raises():
    groups = np.array(["a", "a", "b", "c"])
    from src.models.splitting import SplitIndices

    bad = SplitIndices(train=np.array([0, 1, 2]), val=np.array([2]), test=np.array([3]))
    with pytest.raises(ValueError):
        assert_group_disjoint(groups, bad)


def test_grouped_split_few_groups_warns_not_raises(synthetic_xy):
    _, _, groups = synthetic_xy
    split = grouped_split(groups[:20], val_frac=0.2, test_frac=0.2, seed=1)  # 4 groups
    assert len(split.test) > 0


# ------------------------------------------------------------------ trainers
def test_train_model_each_algorithm_predicts(synthetic_xy):
    X, y, _ = synthetic_xy
    for name in ALGORITHMS:
        model = train_model(name, X, y, seed=42)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)


def test_train_all_returns_all_algorithms(synthetic_xy):
    X, y, _ = synthetic_xy
    models = train_all(X, y, seed=42)
    assert set(models) == set(ALGORITHMS)


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        train_model("nope", np.zeros((10, 2)), np.zeros(10), seed=1)


# ------------------------------------------------------------------- bundles
def _bundle(synthetic_xy):
    X, y, _ = synthetic_xy
    model = train_model("logreg", X, y, seed=42)
    return ModelBundle(
        model=model,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        algorithm="logreg",
        trained_at="2026-10-18T00:00:00",
        n_train=len(y),
        metrics={"f1": 0.9},
    )


def test_save_load_roundtrip(tmp_path, synthetic_xy):
    bundle = _bundle(synthetic_xy)
    X, _, _ = synthetic_xy
    path = save_bundle(bundle, tmp_path / "m.joblib")
    assert path.is_file()
    loaded = load_bundle(path)
    assert loaded.feature_names == bundle.feature_names
    assert loaded.algorithm == "logreg"
    np.testing.assert_array_equal(predict(loaded, X), predict(bundle, X))


def test_load_rejects_wrong_version(tmp_path, synthetic_xy):
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"version": "0", "algorithm": "logreg", "feature_names": ["a"]}, path)
    with pytest.raises(ModelBundleError):
        load_bundle(path)


def test_load_rejects_missing_features(tmp_path, synthetic_xy):
    import joblib

    path = tmp_path / "bad2.joblib"
    joblib.dump({"version": MODEL_VERSION, "algorithm": "logreg", "feature_names": []}, path)
    with pytest.raises(ModelBundleError):
        load_bundle(path)


def test_predict_schema_mismatch_raises(tmp_path, synthetic_xy):
    bundle = _bundle(synthetic_xy)
    X_wrong = np.zeros((5, 99))
    with pytest.raises(ValueError):
        predict(bundle, X_wrong)


# ------------------------------------------------- SoftVoteModel (ensemble)
class StubProbaModel:
    """Fixed P(1) per row; module-level class so it pickles."""

    def __init__(self, proba):
        self.proba = np.asarray(proba, float)

    def predict_proba(self, X):
        p1 = self.proba[: len(X)]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.proba[: len(X)] >= 0.5).astype(int)


def test_soft_vote_model_mean_and_threshold():
    from src.models.tuning import SoftVoteModel

    ens = SoftVoteModel([StubProbaModel([0.9, 0.1]), StubProbaModel([0.2, 0.4])])
    X = np.zeros((2, 3))
    proba = ens.predict_proba(X)
    np.testing.assert_allclose(proba[:, 1], [0.55, 0.25])  # exact column mean
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert list(ens.predict(X)) == [1, 0]  # 0.55 >= 0.5, 0.25 < 0.5
    ens.threshold = 0.6
    assert list(ens.predict(X)) == [0, 0]  # threshold respected


def test_soft_vote_model_validates_members():
    from src.models.tuning import SoftVoteModel

    with pytest.raises(ValueError):
        SoftVoteModel([])


def test_soft_vote_model_joblib_roundtrip(tmp_path, synthetic_xy):
    import joblib

    from src.models.tuning import SoftVoteModel

    X, y, _ = synthetic_xy
    members = [train_model(a, X, y, seed=42) for a in ("logreg", "dt")]
    ens = SoftVoteModel(members, threshold=0.4)
    path = tmp_path / "ens.joblib"
    joblib.dump(ens, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), ens.predict_proba(X))
    assert list(loaded.predict(X)) == list(ens.predict(X))
    assert loaded.threshold == 0.4
