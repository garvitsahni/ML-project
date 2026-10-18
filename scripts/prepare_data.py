"""Build the sliding-window feature table (detect+track+flow+features) -> parquet.

CPU-heavy on the full dataset (~20-50 ms/frame with YOLOv8n). Use
--max-sequences / --max-frames for smoke runs.
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
    from src.datasets.ucsd_anomaly import UcsdAdapter
    from src.features.pipeline import WindowConfig, build_feature_table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output parquet (default: reports/processed/ucsd_windows.parquet)")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames per sequence (smoke runs)")
    args = parser.parse_args()

    paths = load_paths()
    root = args.root or paths.ucsd_dir
    out = args.out or (paths.processed_dir / "ucsd_windows.parquet")
    cfg = WindowConfig(window=args.window, stride=args.stride, grid_size=args.grid)

    try:
        adapter = UcsdAdapter(root=root)
    except Exception as exc:
        print(f"Dataset not available at {root}: {exc}", file=sys.stderr)
        print("Place UCSD v1p2 manually under data/raw/ucsd_anomaly/.", file=sys.stderr)
        return 1

    df = build_feature_table(
        adapter, config=cfg,
        max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames,
        progress=True,
    )
    if df.empty:
        print("No windows produced (dataset too small for the chosen window/stride).",
              file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "rows": int(len(df)),
        "sequences": int(df["sequence_id"].nunique()),
        "window": args.window,
        "stride": args.stride,
        "grid": args.grid,
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "split_sequence_counts": {
            str(k): int(v) for k, v in df.groupby("split")["sequence_id"].nunique().items()
        },
        "table": str(out),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
