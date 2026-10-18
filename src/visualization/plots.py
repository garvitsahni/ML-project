"""Publication-style plots; every function writes a PNG and frees the figure."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_density_heatmap(density_map: np.ndarray, out_path: Path, title: str = "Density estimate") -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(density_map, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path)


def plot_timeseries(
    x: np.ndarray,
    series: Mapping[str, np.ndarray],
    out_path: Path,
    xlabel: str = "frame",
    ylabel: str = "value",
    title: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, y in series.items():
        ax.plot(x, np.asarray(y, dtype=float), label=label, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], out_path: Path, title: str = "Confusion matrix"
) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path)


def plot_feature_importance(
    names: Sequence[str], importances: np.ndarray, out_path: Path, title: str = "Feature importance"
) -> Path:
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in order], np.asarray(importances)[order], color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel("importance")
    return _save(fig, out_path)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)
