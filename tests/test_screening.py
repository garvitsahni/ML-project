"""Tests for the Spearman redundancy screen over schema-v2 candidates."""

import numpy as np
import pandas as pd
import pytest

from src.features.screening import redundant_features


def test_flagged_when_copy_of_existing():
    rng = np.random.default_rng(0)
    base = rng.normal(size=50)
    df = pd.DataFrame({
        "person_count": base,
        "copy_of_count": base * 2.0 + 1.0,   # perfect monotone -> |rho| = 1
        "independent": rng.normal(size=50),
    })
    flagged = redundant_features(df, candidates=["copy_of_count", "independent"],
                                 existing=["person_count"], threshold=0.9)
    assert "copy_of_count" in flagged
    assert "independent" not in flagged


def test_threshold_boundary_respected():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    noisy = x + rng.normal(scale=1.5, size=200)  # rho around 0.5-0.6
    df = pd.DataFrame({"x": x, "noisy": noisy})
    strict = redundant_features(df, ["noisy"], ["x"], threshold=0.3)
    loose = redundant_features(df, ["noisy"], ["x"], threshold=0.9)
    assert "noisy" in strict
    assert "noisy" not in loose


def test_flag_message_names_the_matched_feature():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    flagged = redundant_features(df, ["b"], ["a"])
    assert "a" in flagged["b"]
    assert "rho" in flagged["b"]


def test_candidates_compared_only_to_existing():
    # two candidates identical to each other but not to existing -> neither flagged
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "existing": rng.normal(size=30),
        "cand1": rng.normal(size=30),
        "cand2": None,
    })
    df["cand2"] = df["cand1"]
    flagged = redundant_features(df, ["cand1", "cand2"], ["existing"], threshold=0.9)
    assert flagged == {}
