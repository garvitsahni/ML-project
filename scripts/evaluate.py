"""Evaluate saved bundles on the held-out TEST split; gate Experiment E.

Writes reports/results/evaluation.json (real metrics only) and
reports/results/exp_e_status.json (not_supported without congestion labels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_paths  # noqa: E402


def main() -> int:
    import numpy as np

    from src.experiments.metrics import compute_binary_metrics
    from src.experiments.runner import load_feature_table
    from src.models.bundle import load_bundle
    from src.models.splitting import grouped_split

    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=paths.processed_dir / "ucsd_windows.parquet")
    parser.add_argument("--models-dir", type=Path, default=paths.models_dir)
    parser.add_argument("--results-dir", type=Path, default=paths.results_dir)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        df = load_feature_table(args.table)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Run: python scripts/prepare_data.py first.", file=sys.stderr)
        return 1

    bundles = sorted(args.models_dir.glob("*.joblib")) if args.models_dir.is_dir() else []
    if not bundles:
        print(f"No bundles in {args.models_dir}. Run: python scripts/train_ml.py", file=sys.stderr)
        return 1

    split = grouped_split(df["sequence_id"].to_numpy(), seed=args.seed)
    y = df["label"].to_numpy()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"n_test_windows": int(len(split.test)),
                    "test_sequences": sorted(set(df["sequence_id"].iloc[split.test])),
                    "bundles": {}}
    for path in bundles:
        bundle = load_bundle(path)
        missing = [c for c in bundle.feature_names if c not in df.columns]
        if missing:
            print(f"Skipping {path.name}: table missing {missing}", file=sys.stderr)
            continue
        X = df[bundle.feature_names].iloc[split.test]
        pred = bundle.model.predict(X)
        score = (bundle.model.predict_proba(X)[:, 1]
                 if hasattr(bundle.model, "predict_proba") else None)
        m = compute_binary_metrics(y[split.test], pred, score)
        report["bundles"][bundle.algorithm] = {
            "test_metrics": m, "trained_at": bundle.trained_at, "n_train": bundle.n_train,
            "bundle": path.name,
        }
        print(f"[evaluate] {bundle.algorithm}: test f1={m['f1']:.3f} "
              f"roc_auc={m['roc_auc']} n={m['n_samples']}")

    (args.results_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.results_dir / "exp_e_status.json").write_text(
        json.dumps({
            "status": "not_supported",
            "reason": "No ground-truth congestion labels exist for this dataset; "
                      "Experiment E is intentionally disabled.",
        }, indent=2),
        encoding="utf-8")
    print(f"[evaluate] wrote {args.results_dir / 'evaluation.json'} and exp_e_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
