"""Group-aware splitting: frames from one video/sequence never span splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def grouped_split(
    group_ids: np.ndarray,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Split sample indices by GROUP (e.g. sequence id), never by frame.

    Groups are shuffled with ``seed`` and assigned greedily to
    test -> val -> train so fractions hold at group level. With very few
    groups, each split still receives at least one group when possible.
    """
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be 1-D")
    uniq = np.unique(group_ids)
    if len(uniq) < 2:
        raise ValueError("Need at least 2 distinct groups to split")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:  # keep at least one train group
        overflow = n_test + n_val - (n - 1)
        n_val = max(1, n_val - overflow)
        n_test = max(1, min(n_test, n - n_val - 1)) if n - n_val - 1 >= 1 else 1

    test_groups = set(uniq[:n_test])
    val_groups = set(uniq[n_test:n_test + n_val])
    train_groups = set(uniq[n_test + n_val:])

    def idx_for(groups: set) -> np.ndarray:
        return np.array([i for i, g in enumerate(group_ids) if g in groups], dtype=int)

    split = SplitIndices(
        train=idx_for(train_groups),
        val=idx_for(val_groups),
        test=idx_for(test_groups),
    )
    if len(split.train) == 0:
        raise ValueError("Split produced an empty train set; adjust fractions or add groups")
    return split


def assert_group_disjoint(group_ids: np.ndarray, split: SplitIndices) -> None:
    """Raise ValueError if any group appears in more than one split."""
    group_ids = np.asarray(group_ids)
    sets = [
        set(group_ids[split.train].tolist()),
        set(group_ids[split.val].tolist()),
        set(group_ids[split.test].tolist()),
    ]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(f"Group leakage between splits: {sorted(overlap)[:5]}")
