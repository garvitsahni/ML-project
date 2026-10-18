"""Shared dataset types for CrowdSense AI adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class MissingDatasetError(FileNotFoundError):
    """Raised when a dataset root or required annotation is absent."""


@dataclass(frozen=True)
class InspectionReport:
    """Result of inspecting a dataset without loading frames.

    Never fabricates values: counts reflect what is actually on disk.
    """

    dataset: str
    root: Path
    image_count: int
    annotation_count: int
    images_missing: list[str] = field(default_factory=list)
    annotations_missing: list[str] = field(default_factory=list)
    labels_available: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
