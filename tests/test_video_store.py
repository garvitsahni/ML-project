"""video_store: one temp upload file, hash-keyed, replaced on new upload."""

from src.ui import video_store


def test_save_upload_writes_bytes_and_reuses_path_per_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(video_store, "_upload_dir", lambda: tmp_path)
    p1 = video_store.save_upload("abc123", b"AAAA", suffix=".mp4")
    assert p1.exists()
    assert p1.read_bytes() == b"AAAA"
    assert p1.suffix == ".mp4"
    p2 = video_store.save_upload("abc123", b"BBBB", suffix=".mp4")
    assert p2 == p1  # same hash -> same path
    assert p1.read_bytes() == b"BBBB"


def test_save_upload_keeps_only_one_active_file(tmp_path, monkeypatch):
    monkeypatch.setattr(video_store, "_upload_dir", lambda: tmp_path)
    p1 = video_store.save_upload("h1", b"A", suffix=".mp4")
    p2 = video_store.save_upload("h2", b"B", suffix=".avi")
    assert p2.exists() and p2.suffix == ".avi"
    assert not p1.exists()  # replaced, no stale temp files


def test_save_upload_sanitizes_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(video_store, "_upload_dir", lambda: tmp_path)
    path = video_store.save_upload("../../evil/../name", b"X", suffix=".mp4")
    assert path.parent == tmp_path
    assert path.name.count("..") == 0
    assert "/" not in path.name and "\\" not in path.name


def test_cleanup_upload_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(video_store, "_upload_dir", lambda: tmp_path)
    video_store.save_upload("h1", b"A", suffix=".mp4")
    video_store.cleanup_upload()
    video_store.cleanup_upload()  # no error when nothing exists
    assert list(tmp_path.glob("*")) == []
