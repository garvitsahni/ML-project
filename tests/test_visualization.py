"""Tests for visualization helpers (Agg backend, real PNG output)."""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.visualization.plots import (
    plot_confusion_matrix,
    plot_density_heatmap,
    plot_feature_importance,
    plot_roc,
    plot_timeseries,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_density_heatmap_writes_png(tmp_path):
    p = plot_density_heatmap(np.random.rand(48, 64).astype(np.float32), tmp_path / "d.png")
    assert p.is_file() and p.stat().st_size > 0
    assert plt.get_fignums() == []  # figure released


def test_timeseries_writes_png(tmp_path):
    t = np.arange(10)
    p = plot_timeseries(t, {"count": np.arange(10.0), "speed": np.linspace(0, 1, 10)},
                        tmp_path / "ts.png", xlabel="frame", ylabel="value", title="T")
    assert p.is_file() and p.stat().st_size > 0


def test_confusion_matrix_writes_png(tmp_path):
    p = plot_confusion_matrix(np.array([[8, 2], [1, 9]]), ["normal", "anomaly"], tmp_path / "cm.png")
    assert p.is_file() and p.stat().st_size > 0


def test_feature_importance_writes_png(tmp_path):
    p = plot_feature_importance(["a", "b", "c"], np.array([0.5, 0.3, 0.2]), tmp_path / "fi.png")
    assert p.is_file() and p.stat().st_size > 0


def test_roc_writes_png(tmp_path):
    fpr = np.linspace(0, 1, 20)
    tpr = np.sqrt(fpr)
    p = plot_roc(fpr, tpr, 0.87, tmp_path / "roc.png")
    assert p.is_file() and p.stat().st_size > 0
