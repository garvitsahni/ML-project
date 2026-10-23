"""Temp-file store for uploaded videos — never accumulates files.

A single upload file is kept alive at a time (hash-keyed path), so the
temp directory cannot pile up videos across sessions. The file is removed
by ``cleanup_upload()`` from the page's post-run lifecycle.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

_UPLOAD_SUBDIR = "crowdsense_video_upload"
_MAX_HASH_CHARS = 40


def _upload_dir() -> Path:
    """Directory holding the active upload (monkeypatchable in tests)."""
    return Path(tempfile.gettempdir()) / _UPLOAD_SUBDIR


def _safe_hash(content_hash: str) -> str:
    """Sanitize a content hash for use in a filename (no traversal)."""
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(content_hash))[:_MAX_HASH_CHARS]
    return cleaned or "upload"


def save_upload(content_hash: str, data: bytes, suffix: str = ".mp4") -> Path:
    """Write the upload under a hash-keyed path, replacing any previous file.

    Only one upload file is kept alive, so re-submitting or switching files
    never leaves stale videos on disk. Returns the written path.
    """
    directory = _upload_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.iterdir():
        if old.is_file():
            old.unlink(missing_ok=True)
    path = directory / f"upload_{_safe_hash(content_hash)}{suffix}"
    path.write_bytes(data)
    return path


def cleanup_upload() -> None:
    """Remove the active upload file (idempotent, no error when empty)."""
    directory = _upload_dir()
    if directory.is_dir():
        for old in directory.iterdir():
            if old.is_file():
                old.unlink(missing_ok=True)
