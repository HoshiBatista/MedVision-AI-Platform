"""
Test fixtures for report_service.

The generator is a thin Ollama HTTP client; tests mock it so no Ollama server or
model download is needed. Runs on a throwaway SQLite DB.
"""

import asyncio
import contextlib
import os
import sys
import tempfile
from collections.abc import Iterator

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="report_test_")
os.close(_DB_FD)
os.environ.update(DATABASE_URL=f"sqlite+aiosqlite:///{_DB_PATH}", ENVIRONMENT="test")

import pytest  # noqa: E402
from app.core.database import Base, engine  # noqa: E402
from app.services.report_generator import report_generator  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FAKE_CONTENT = "FINDINGS: a representative finding. IMPRESSION: stable."


async def _drop_all() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def pytest_sessionfinish() -> None:
    with contextlib.suppress(OSError):
        os.remove(_DB_PATH)


@pytest.fixture
def client(monkeypatch) -> Iterator[TestClient]:
    asyncio.run(_drop_all())

    async def _fake_generate(prompt: str) -> str:
        return f"{prompt[:20]} {FAKE_CONTENT}"

    async def _noop_ensure() -> None:
        return None

    monkeypatch.setattr(report_generator, "ensure_model", _noop_ensure)
    monkeypatch.setattr(report_generator, "is_ready", lambda: True)
    monkeypatch.setattr(report_generator, "generate", _fake_generate)

    from app.main import app

    with TestClient(app) as c:
        yield c
