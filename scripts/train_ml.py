"""Train ML bundles + run experiments A-D on a feature table.

Bundles are trained on the TRAIN split only and saved per algorithm;
experiment metrics are computed on the held-out TEST split (grouped by
sequence id — never random frames).

With --tune, the four sanctioned evaluations happen inside run_attribution
(the ONLY test-touching path); tuned bundles are then refit on train with
the per-algorithm val threshold, plus an optional soft-vote ensemble
bundle. --robustness runs the val-only label-robustness experiment.
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
    from src.experiments.attribution import run_attribution
    from src.experiments.runner import (
        ExperimentConfig,
        ablation_experiment,
        algorithm_comparison,
        label_robustness_experiment,
        load_feature_table,
    )
    from src.features.schema import FeatureSchema
    from src.models.bundle import ModelBundle, now_iso, save_bundle
    from src.models.splitting import grouped_split
    from src.models.trainers import ALGORITHMS, train_model
    from src.models.tuning import SoftVoteModel, tune_model

    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=paths.processed_dir / "ucsd_windows.parquet")
    parser.add_argument("--algorithms", type=str, default=",".join(ALGORITHMS),
                        help="Comma-separated subset (default: all five)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=paths.results_dir)
    parser.add_argument("--figures-dir", type=Path, default=paths.figures_dir)
    parser.add_argument("--models-dir", type=Path, default=paths.models_dir)
    parser.add_argument("--tune", action="store_true",
                        help="Grouped-CV tuning + val thresholds + ensemble; "
                             "test metrics come from the sanctioned attribution run")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="With --tune: skip the soft-vote ensemble bundle")
    parser.add_argument("--robustness", action="store_true",
                        help="Also run the val-only label-robustness experiment")
    parser.add_argument("--exclude-features", type=str, default="",
                        help="Comma-separated schema features to drop "
                             "(from screen_features.py)")
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

    drop = [c.strip() for c in args.exclude_features.split(",") if c.strip()]
    if drop:
        missing = [c for c in drop if c not in df.columns]
        if missing:
            print(f"--exclude-features not in table: {missing}", file=sys.stderr)
            return 1
        df = df.drop(columns=drop)
        print(f"[train] excluded features: {drop}")

    config = ExperimentConfig(results_dir=args.results_dir, figures_dir=args.figures_dir,
                              seed=args.seed, algorithms=algos)
    print(f"[train] {len(df)} windows, {df['sequence_id'].nunique()} sequences")
    if args.robustness:
        label_robustness_experiment(df, config)
        print("[train] label robustness done ->",
              args.results_dir / "experiment_label_robustness.json")
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
    groups = df["sequence_id"].to_numpy()
    split = grouped_split(groups, seed=args.seed)
    y = df["label"].to_numpy()
    X = df[cols]
    args.models_dir.mkdir(parents=True, exist_ok=True)
    from src.experiments.metrics import compute_binary_metrics

    if args.tune:
        # one sanctioned 4-config evaluation; everything below reuses its outputs
        attribution = run_attribution(df, seed=args.seed, results_dir=args.results_dir)
        per_algo = attribution["tuned_threshold"]["per_algorithm"]
        (args.results_dir / "experiment_tuning.json").write_text(
            json.dumps(per_algo, indent=2, default=str), encoding="utf-8")
        for algo in algos:
            if algo not in per_algo:
                print(f"[train] no tuned entry for {algo}; skipped", file=sys.stderr)
                continue
            entry = per_algo[algo]
            model, _params, _cv = tune_model(
                algo, X.iloc[split.train], y[split.train], groups[split.train],
                seed=args.seed)
            bundle = ModelBundle(
                model=model, feature_names=list(cols), algorithm=algo,
                trained_at=now_iso(), n_train=int(len(split.train)),
                metrics=entry["test_metrics"], threshold=float(entry["threshold"]),
                schema_fingerprint=schema.fingerprint,
            )
            path = save_bundle(bundle, args.models_dir / f"{algo}.joblib")
            print(f"[train] saved tuned {path} (threshold={entry['threshold']}, "
                  f"test f1={entry['test_metrics'].get('f1')})")
        if not args.no_ensemble:
            final = attribution["ensemble_final"]
            member_models = [
                tune_model(a, X.iloc[split.train], y[split.train], groups[split.train],
                           seed=args.seed)[0]
                for a in final["members"]
            ]
            ens = SoftVoteModel(member_models, threshold=float(final["threshold"]))
            bundle = ModelBundle(
                model=ens, feature_names=list(cols), algorithm="ensemble",
                trained_at=now_iso(), n_train=int(len(split.train)),
                metrics=final["test_metrics"], threshold=float(final["threshold"]),
                schema_fingerprint=schema.fingerprint,
            )
            path = save_bundle(bundle, args.models_dir / "ensemble.joblib")
            ensemble_json = {
                "members": [
                    {"algorithm": a,
                     "val_auc": per_algo[a]["val_metrics"].get("roc_auc")}
                    for a in final["members"]
                ],
                "threshold": float(final["threshold"]),
                "test_metrics": final["test_metrics"],
                "test_sequence_ids": final["test_sequence_ids"],
            }
            (args.results_dir / "experiment_ensemble.json").write_text(
                json.dumps(ensemble_json, indent=2, default=str), encoding="utf-8")
            print(f"[train] saved ensemble {path} (members={final['members']}, "
                  f"test f1={final['test_metrics'].get('f1')})")
        print("[train] complete.")
        return 0

    for algo in algos:
        model = train_model(algo, X.iloc[split.train], y[split.train], seed=args.seed)
        test_metrics = {}
        if len(split.test):
            pred = model.predict(X.iloc[split.test])
            score = model.predict_proba(X.iloc[split.test])[:, 1] if hasattr(model, "predict_proba") else None
            test_metrics = compute_binary_metrics(y[split.test], pred, score)
        bundle = ModelBundle(
            model=model, feature_names=list(cols), algorithm=algo,
            trained_at=now_iso(), n_train=int(len(split.train)), metrics=test_metrics,
            schema_fingerprint=schema.fingerprint,
        )
        path = save_bundle(bundle, args.models_dir / f"{algo}.joblib")
        print(f"[train] saved {path} (train={len(split.train)} windows, "
              f"test f1={test_metrics.get('f1')})")
    print("[train] complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
