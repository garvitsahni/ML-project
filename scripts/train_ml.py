"""Train ML bundles + run experiments A-D on a feature table.

Bundles are trained on the TRAIN split only and saved per algorithm;
experiment metrics are computed on the held-out TEST split (grouped by
sequence id — never random frames).
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

    from src.experiments.runner import ExperimentConfig, ablation_experiment, algorithm_comparison, load_feature_table
    from src.features.schema import FeatureSchema
    from src.models.bundle import ModelBundle, now_iso, save_bundle
    from src.models.splitting import grouped_split
    from src.models.trainers import ALGORITHMS, train_model

    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=paths.processed_dir / "ucsd_windows.parquet")
    parser.add_argument("--algorithms", type=str, default=",".join(ALGORITHMS),
                        help="Comma-separated subset (default: all five)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=paths.results_dir)
    parser.add_argument("--figures-dir", type=Path, default=paths.figures_dir)
    parser.add_argument("--models-dir", type=Path, default=paths.models_dir)
    args = parser.parse_args()

    algos = tuple(a.strip() for a in args.algorithms.split(",") if a.strip())
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        print(f"Unknown algorithms: {unknown}. Available: {list(ALGORITHMS)}", file=sys.stderr)
        return 1

    try:
        df = load_feature_table(args.table)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Run: python scripts/prepare_data.py", file=sys.stderr)
        return 1

    config = ExperimentConfig(results_dir=args.results_dir, figures_dir=args.figures_dir,
                              seed=args.seed, algorithms=algos)
    print(f"[train] {len(df)} windows, {df['sequence_id'].nunique()} sequences")
    abl = ablation_experiment(df, config)
    print("[train] ablation A/B/C done ->", args.results_dir / "experiment_ablation.json")
    cmp_ = algorithm_comparison(df, config)
    print("[train] algorithm comparison D done ->", args.results_dir / "experiment_algorithms.json")

    schema = FeatureSchema.default()
    cols = [c for c in schema.names if c in df.columns]
    missing = [c for c in schema.names if c not in df.columns]
    if missing:
        print(f"Table missing schema features: {missing}", file=sys.stderr)
        return 1
    split = grouped_split(df["sequence_id"].to_numpy(), seed=args.seed)
    y = df["label"].to_numpy()
    X = df[cols]
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for algo in algos:
        model = train_model(algo, X.iloc[split.train], y[split.train], seed=args.seed)
        test_metrics = {}
        if len(split.test):
            pred = model.predict(X.iloc[split.test])
            from src.experiments.metrics import compute_binary_metrics

            score = model.predict_proba(X.iloc[split.test])[:, 1] if hasattr(model, "predict_proba") else None
            test_metrics = compute_binary_metrics(y[split.test], pred, score)
        bundle = ModelBundle(
            model=model, feature_names=list(cols), algorithm=algo,
            trained_at=now_iso(), n_train=int(len(split.train)), metrics=test_metrics,
        )
        path = save_bundle(bundle, args.models_dir / f"{algo}.joblib")
        print(f"[train] saved {path} (train={len(split.train)} windows, "
              f"test f1={test_metrics.get('f1')})")
    print("[train] complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
