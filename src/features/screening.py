"""Spearman redundancy screen for schema-v2 candidate features.

Flags candidates whose absolute Spearman correlation with any EXISTING
feature exceeds a threshold — candidates are never compared to each
other (the model may legitimately use two correlated new features; the
risk screen is about duplicating what the old schema already carries).
"""

from __future__ import annotations

import pandas as pd


def redundant_features(
    df: pd.DataFrame,
    candidates: list[str],
    existing: list[str],
    threshold: float = 0.9,
) -> dict[str, str]:
    """{candidate: "matched:<existing> (|rho|=<value:.2f>)"} for flagged ones."""
    flagged: dict[str, str] = {}
    existing_cols = [c for c in existing if c in df.columns]
    for cand in candidates:
        if cand not in df.columns:
            continue
        best_col, best_rho = None, 0.0
        for col in existing_cols:
            rho = df[cand].corr(df[col], method="spearman")
            if pd.notna(rho) and abs(rho) > best_rho:
                best_col, best_rho = col, abs(rho)
        if best_col is not None and best_rho > threshold:
            flagged[cand] = f"matched:{best_col} (|rho|={best_rho:.2f})"
    return flagged
