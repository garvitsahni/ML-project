"""Calibrate the person detector on real UCSD train frames (honest recall proxy).

Runs a conf x imgsz grid over a deterministic sample of TRAIN-split frames,
records mean detections/frame and timing, saves overlay images for audit,
and selects the config by the pre-declared rule: highest mean detections
per frame among configs whose projected full rebuild fits --max-hours
(ties break toward the faster config).

Detections are DETECTOR ESTIMATES — this project has no counting ground
truth; the calibration optimizes feature-quality recall, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_paths  # noqa: E402


def main() -> int:
    import cv2
    import numpy as np

    from src.detection.calibration import (
        project_rebuild_hours,
        sample_frame_files,
        select_config,
    )
    from src.detection.detector import PersonDetector, draw_detections
    from src.preprocessing.video import read_frame

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="UCSD dataset root (default: configs/paths.yaml)")
    parser.add_argument("--conf-list", type=str, default="0.05,0.10,0.15,0.25")
    parser.add_argument("--imgsz-list", type=str, default="320,640,1280")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports" / "detection_calibration")
    args = parser.parse_args()

    from src.datasets.ucsd_anomaly import UcsdAdapter

    paths = load_paths()
    try:
        adapter = UcsdAdapter(root=args.root or paths.ucsd_dir)
        frame_files = sample_frame_files(adapter, n_frames=args.max_frames)
    except Exception as exc:
        print(f"Dataset not available: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    confs = [float(c) for c in args.conf_list.split(",") if c.strip()]
    imgszs = [int(s) for s in args.imgsz_list.split(",") if s.strip()]

    results = []
    for conf in confs:
        for imgsz in imgszs:
            detector = PersonDetector(conf=conf, imgsz=imgsz)
            counts, times = [], []
            overlays = []
            for i, fpath in enumerate(frame_files):
                frame = read_frame(fpath)
                t0 = time.perf_counter()
                res = detector.detect(frame)
                times.append(time.perf_counter() - t0)
                counts.append(len(res.detections))
                if i < 3:  # first 3 frames overlaid per config for audit
                    tag = f"conf{conf:.2f}_imgsz{imgsz}_frame{i}.png"
                    cv2.imwrite(str(out / tag), draw_detections(frame, res.detections))
                    overlays.append(tag)
            mean_s = float(np.mean(times))
            results.append({
                "conf": conf,
                "imgsz": imgsz,
                "mean_detections_per_frame": float(np.mean(counts)),
                "mean_seconds_per_frame": mean_s,
                "projected_rebuild_hours": project_rebuild_hours(mean_s),
                "overlay_files": overlays,
            })
            print(f"[calibrate] conf={conf:.2f} imgsz={imgsz}: "
                  f"{np.mean(counts):.2f} dets/frame, {mean_s*1000:.0f} ms/frame, "
                  f"projects {project_rebuild_hours(mean_s):.2f}h rebuild")

    try:
        selected = select_config(results, max_rebuild_hours=args.max_hours)
    except ValueError as exc:
        print(f"No config selected: {exc}", file=sys.stderr)
        return 1

    payload = {
        "results": results,
        "selected": selected,
        "selection_rule": (
            "highest mean_detections_per_frame among configs projecting a full "
            f"rebuild <= {args.max_hours}h; ties break toward faster config"
        ),
        "n_frames_sampled": len(frame_files),
        "honesty": "detections are detector estimates; no counting ground truth exists",
    }
    (out / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calibrate] selected conf={selected['conf']} imgsz={selected['imgsz']} "
          f"-> {out / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
