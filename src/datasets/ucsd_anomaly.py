"""Adapter for the UCSD Pedestrian Anomaly Detection Dataset (v1p2).

Layout (verified in this workspace)::

    UCSD_Anomaly_Dataset.v1p2/
      UCSDped1/
        Train/Train001/001.tif ...     (34 train sequences, frame folders)
        Test/Test001/001.tif ...       (36 test sequences)
        Test/Test001_gt/003.bmp ...    (pixel masks for 10 test clips)
        Test/UCSDped1.m                (frame-level abnormal-event GT)
      UCSDped2/                        (16 train / 12 test, same shape)

The ``.m`` files list, per test sequence (in order Test001, Test002, ...),
the 1-based frame numbers containing abnormal events::

    TestVideoFile{end+1}.gt_frame = [60:152];
    TestVideoFile{end+1}.gt_frame = [5:90, 140:200];

Per the dataset README: training clips contain ONLY normal frames; every
test clip contains at least some anomalous frames.

Citation: Mahadevan, Vallabhajosyula, Vasconcelos; "Anomaly Detection in
Crowded Scenes"; CVPR 2010. Dataset provided manually by the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from src.config import load_paths
from src.datasets.base import InspectionReport, MissingDatasetError

# A single `.m` entry: [3:5] or [1:2, 140:200]
_GT_ENTRY_RE = re.compile(
    r"gt_frame\s*=\s*\[([^\]]*)\]", re.IGNORECASE
)
_RANGE_RE = re.compile(r"(\d+)\s*:\s*(\d+)")
_FRAME_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Expected counts from the dataset README (used only as informational notes).
_README_COUNTS = {"UCSDped1": {"train": 34, "test": 36}, "UCSDped2": {"train": 16, "test": 12}}


class AnnotationError(RuntimeError):
    """Raised when annotations are missing or inconsistent with the data."""


@dataclass(frozen=True)
class AnomalySequence:
    """One UCSD sequence (a folder of frames) with per-frame anomaly labels."""

    sequence_id: str  # e.g. "ped1_Test001"
    part: str  # "ped1" | "ped2"
    split: str  # "train" | "test"
    frames_dir: Path
    frame_files: tuple[Path, ...]
    n_frames: int
    frame_labels: np.ndarray  # shape (n_frames,), int 0/1, aligned to frame_files
    has_masks: bool  # a sibling *_gt folder with .bmp masks exists


def parse_gt_frames(text: str) -> list[frozenset[int]]:
    """Parse a UCSD ``.m`` annotation file into per-test-sequence frame sets.

    Returns one frozenset of 1-based anomalous frame numbers per
    ``gt_frame`` entry, in file order (entry i = Test sequence i+1).
    """
    entries: list[frozenset[int]] = []
    for match in _GT_ENTRY_RE.finditer(text):
        body = match.group(1)
        frames: set[int] = set()
        for rng in _RANGE_RE.finditer(body):
            start, end = int(rng.group(1)), int(rng.group(2))
            frames.update(range(start, end + 1))
        # tolerate comma lists of single numbers too (rare)
        if not _RANGE_RE.search(body):
            for token in body.split(","):
                token = token.strip()
                if token.isdigit():
                    frames.add(int(token))
        entries.append(frozenset(frames))
    return entries


def _is_junk(path: Path) -> bool:
    """Skip macOS AppleDouble files (``._foo.tif``) and non-frame dirs."""
    return path.name.startswith("._") or path.name.startswith(".")


class UcsdAdapter:
    """Enumerate UCSD sequences and their frame-level anomaly labels."""

    name = "ucsd_anomaly"

    def __init__(self, root: Path | None = None):
        self._root_override = root

    @property
    def root(self) -> Path:
        return self._root_override if self._root_override is not None else load_paths().ucsd_dir

    # ------------------------------------------------------------------ parts
    def _parts(self, root: Path) -> list[Path]:
        parts = sorted(
            d
            for d in root.glob("UCSDped*")
            if d.is_dir() and not _is_junk(d)
        )
        if not parts:
            raise MissingDatasetError(
                f"No UCSDped* parts found under {root}. Expected UCSDped1/ and UCSDped2/."
            )
        return parts

    @staticmethod
    def _part_name(part_dir: Path) -> str:  # "UCSDped1" -> "ped1"
        return part_dir.name.replace("UCSD", "").lower()

    def _sequence_dirs(self, part_dir: Path, split: str) -> list[Path]:
        split_dir = part_dir / split.capitalize()
        if not split_dir.is_dir():
            return []
        dirs = [
            d
            for d in split_dir.iterdir()
            if d.is_dir()
            and not _is_junk(d)
            and not d.name.endswith("_gt")
            and d.name.lower().startswith(split.lower())
        ]
        return sorted(dirs, key=lambda d: d.name)

    @staticmethod
    def _frame_files(seq_dir: Path) -> tuple[Path, ...]:
        frames = [f for f in seq_dir.iterdir() if f.suffix.lower() in _FRAME_EXTS and not _is_junk(f)]
        return tuple(sorted(frames, key=lambda f: f.name))

    def _gt_entries(self, part_dir: Path) -> dict[str, list[frozenset[int]]]:
        """Parse Test/<part>.m -> {part_name: [frame sets per test sequence]}."""
        part = self._part_name(part_dir)
        gt_path = part_dir / "Test" / f"{part_dir.name}.m"
        if not gt_path.is_file():
            return {part: []}
        text = gt_path.read_text(encoding="utf-8", errors="replace")
        return {part: parse_gt_frames(text)}

    # ----------------------------------------------------------------- public
    def inspect(self, root: Path | None = None) -> InspectionReport:
        root = root if root is not None else self.root
        if not root.is_dir():
            raise MissingDatasetError(
                f"UCSD dataset not found at {root}. "
                "Place UCSD_Anomaly_Dataset.v1p2 there (see README Datasets section)."
            )

        image_count = 0
        annotation_count = 0
        images_missing: list[str] = []
        annotations_missing: list[str] = []
        notes: list[str] = []

        for part_dir in self._parts(root):
            part = self._part_name(part_dir)
            gt_entries = self._gt_entries(part_dir)[part]
            if not gt_entries:
                annotations_missing.append(str(part_dir / "Test" / f"{part_dir.name}.m"))

            for split in ("train", "test"):
                seq_dirs = self._sequence_dirs(part_dir, split)
                expected = _README_COUNTS.get(part_dir.name, {}).get(split)
                if expected is not None and len(seq_dirs) != expected:
                    notes.append(
                        f"{part} {split}: found {len(seq_dirs)} sequences, README says {expected}"
                    )
                for i, seq_dir in enumerate(seq_dirs):
                    frames = self._frame_files(seq_dir)
                    image_count += len(frames)
                    if not frames:
                        images_missing.append(f"{part}/{seq_dir.name}: no frames")
                    if split == "test" and gt_entries:
                        if i < len(gt_entries):
                            out_of_range = sorted(f for f in gt_entries[i] if f > len(frames))
                            for f in out_of_range:
                                images_missing.append(
                                    f"{part}/{seq_dir.name}: gt frame {f} exceeds n_frames={len(frames)}"
                                )
                            annotation_count += len(gt_entries[i])
                        else:
                            notes.append(
                                f"{part}: {len(gt_entries)} gt entries but {len(seq_dirs)} test sequences; "
                                f"extra sequences have no labels"
                            )
        if any("gt" in m for m in images_missing):
            notes.append("Some GT frames exceed actual frame counts (dataset inconsistency).")

        return InspectionReport(
            dataset=self.name,
            root=root,
            image_count=image_count,
            annotation_count=annotation_count,
            images_missing=sorted(images_missing),
            annotations_missing=sorted(annotations_missing),
            labels_available=["frame_level_test_gt"] if not annotations_missing else [],
            notes=notes,
        )

    def iter_sequences(
        self,
        root: Path | None = None,
        split: str | None = None,
        part: str | None = None,
    ) -> Iterator[AnomalySequence]:
        root = root if root is not None else self.root
        if not root.is_dir():
            raise MissingDatasetError(
                f"UCSD dataset not found at {root}. "
                "Place UCSD_Anomaly_Dataset.v1p2 there (see README Datasets section)."
            )

        for part_dir in self._parts(root):
            part_name = self._part_name(part_dir)
            if part is not None and part_name != part:
                continue
            gt_map = self._gt_entries(part_dir)
            gt_entries = gt_map[part_name]

            for split_name in ("train", "test"):
                if split is not None and split_name != split:
                    continue
                seq_dirs = self._sequence_dirs(part_dir, split_name)
                if split_name == "test":
                    if not gt_entries:
                        raise AnnotationError(
                            f"Missing or empty frame-level GT file "
                            f"{part_dir / 'Test' / (part_dir.name + '.m')}"
                        )
                    if len(gt_entries) != len(seq_dirs):
                        raise AnnotationError(
                            f"{part_name}: {len(gt_entries)} gt_frame entries vs "
                            f"{len(seq_dirs)} test sequences — order cannot be aligned."
                        )

                for i, seq_dir in enumerate(seq_dirs):
                    frames = self._frame_files(seq_dir)
                    if split_name == "train":
                        labels = np.zeros(len(frames), dtype=np.int64)
                    else:
                        anomalous = gt_entries[i]
                        labels = np.array(
                            [1 if (fi + 1) in anomalous else 0 for fi in range(len(frames))],
                            dtype=np.int64,
                        )
                    masks_dir = seq_dir.parent / f"{seq_dir.name}_gt"
                    yield AnomalySequence(
                        sequence_id=f"{part_name}_{seq_dir.name}",
                        part=part_name,
                        split=split_name,
                        frames_dir=seq_dir,
                        frame_files=frames,
                        n_frames=len(frames),
                        frame_labels=labels,
                        has_masks=masks_dir.is_dir(),
                    )
