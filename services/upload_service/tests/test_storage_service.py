"""Unit tests for the local-disk storage helper."""

from pathlib import Path

from app.core.config import settings
from app.services import storage_service


async def test_save_file_writes_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))

    path = await storage_service.save_file("study-1", "original.png", b"hello-bytes")

    saved = Path(path)
    assert saved.exists()
    assert saved.read_bytes() == b"hello-bytes"
    assert saved.parent.name == "study-1"


async def test_save_file_creates_nested_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_root", str(tmp_path / "nested" / "root"))
    path = await storage_service.save_file("abc", "original.dcm", b"x")
    assert Path(path).exists()


def test_get_file_path_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))
    p = storage_service.get_file_path("study-42", "original.jpg")
    assert p == tmp_path / "study-42" / "original.jpg"
