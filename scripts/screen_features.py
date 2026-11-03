"""Screen schema-v2 candidate features for redundancy against the original 15.

Run AFTER a feature-table rebuild. Prints candidates to exclude via
`train_ml.py --exclude-features` and writes the full report as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V2_CANDIDATES = [
    "nn_distance_mean", "nn_distance_std", "abrupt_stop_fraction",
    "direction_reversal_rate", "fast_mover_fraction", "speed_p90",
    "track_turnover_rate", "flow_direction_inconsistency",
]

from src.features.schema import ORIGINAL_15  # noqa: E402  (after sys.path setup)


def train_val_subset(df):
    """Drop the sanctioned test split before screening (train/val only).

    Uses the project's grouped sequence split (seed 42) — the same one the
    experiments use — so no test-split row ever feeds feature selection.
    """
    import numpy as np

    from src.models.splitting import grouped_split

    split = grouped_split(df["sequence_id"].to_numpy(), seed=42)
    return df.iloc[np.concatenate([split.train, split.val])]


def main() -> int:
    import pandas as pd

    from src.features.screening import redundant_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path,
                        default=ROOT / "data" / "processed" / "ucsd_windows.parquet")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports" / "results" / "feature_screen.json")
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    if not args.table.is_file():
        print(f"Feature table not found: {args.table}. Run prepare_data.py first.",
              file=sys.stderr)
        return 1

    df = train_val_subset(pd.read_parquet(args.table))
    per_feature = {}
    for cand in V2_CANDIDATES:
        if cand not in df.columns:
            continue
        best_col, best_rho = None, 0.0
        for col in ORIGINAL_15:
            if col not in df.columns:
                continue
            rho = df[cand].corr(df[col], method="spearman")
            if pd.notna(rho) and abs(rho) > best_rho:
                best_col, best_rho = col, abs(rho)
        per_feature[cand] = {
            "max_abs_spearman": None if best_col is None else round(float(best_rho), 4),
            "most_correlated_with": best_col,
        }

    flagged = redundant_features(df, V2_CANDIDATES, ORIGINAL_15, threshold=args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": args.threshold,
        "per_feature": per_feature,
        "flagged": flagged,
        "exclude_features_flag": ",".join(flagged),
        "note": "flagged candidates are excluded from bundle feature sets via "
                "train_ml.py --exclude-features; they stay in the schema/table; "
                "screened on train+val rows only (grouped_split seed=42, "
                "test sequences excluded)",
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if flagged:
        print("Redundant candidates (exclude via --exclude-features):")
        for name, why in flagged.items():
            print(f"  {name}: {why}")
    else:
        print("No candidate exceeded the redundancy threshold.")
    print(f"[screen] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
