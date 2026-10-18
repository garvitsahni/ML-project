"""Inspect a dataset (UCSD) via its adapter; graceful when absent. Read-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.config import load_paths
    from src.datasets.ucsd_anomaly import UcsdAdapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="Dataset root (default: configs/paths.yaml)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to write a JSON summary")
    args = parser.parse_args()

    root = args.root or load_paths().ucsd_dir
    try:
        adapter = UcsdAdapter(root=root)
        seqs = list(adapter.iter_sequences())
    except Exception as exc:
        print(f"Dataset not found at {root}: {exc}", file=sys.stderr)
        print("Place UCSD v1p2 manually under data/raw/ucsd_anomaly/ "
              "(never auto-downloaded).", file=sys.stderr)
        return 1

    n_frames = sum(s.n_frames for s in seqs)
    n_anom = sum(int(s.frame_labels.sum()) for s in seqs if s.frame_labels is not None)
    by_part = {}
    for s in seqs:
        by_part.setdefault(s.part, {"train": 0, "test": 0})[s.split] += 1
    summary = {
        "root": str(root),
        "sequences": len(seqs),
        "frames": n_frames,
        "anomalous_frame_labels": n_anom,
        "sequences_by_part_split": by_part,
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
