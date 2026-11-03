"""Run the four pre-declared attribution evaluations on a feature table.

The ONLY sanctioned test-split evaluation beyond the A-D suite (spec
Goal 3): baseline (old 15 features), schema v2 untuned, tuned+threshold,
and the soft-vote ensemble — all logged with provenance to
reports/results/experiment_attribution.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import pandas as pd

    from src.config import load_paths
    from src.experiments.attribution import run_attribution
    from src.experiments.runner import load_feature_table

    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path,
                        default=paths.processed_dir / "ucsd_windows.parquet")
    parser.add_argument("--results-dir", type=Path, default=paths.results_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-features", type=str, default="",
                        help="Comma-separated schema-v2 features to drop "
                             "(from screen_features.py)")
    args = parser.parse_args()

    try:
        df = load_feature_table(args.table)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Run: python scripts/prepare_data.py", file=sys.stderr)
        return 1

    drop = [c.strip() for c in args.exclude_features.split(",") if c.strip()]
    missing = [c for c in drop if c not in df.columns]
    if missing:
        print(f"--exclude-features not in table: {missing}", file=sys.stderr)
        return 1
    if drop:
        df = df.drop(columns=drop)
        print(f"[attribution] excluded features: {drop}")

    result = run_attribution(df, seed=args.seed, results_dir=args.results_dir)
    for cfg, payload in result.items():
        tm = payload["test_metrics"]
        print(f"[attribution] {cfg}: test AUC={tm.get('roc_auc')} F1={tm.get('f1')}")
    print(f"[attribution] provenance -> {args.results_dir / 'experiment_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
