"""
Test fixtures for report_service.

torch/transformers are stubbed in sys.modules so importing the app never pulls
the heavy ML stack or downloads BioGPT — the generator itself is mocked. Runs
on a throwaway SQLite DB.
"""

import asyncio
import contextlib
import os
import sys
import tempfile
import types
from collections.abc import Iterator

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

# ── Stub the heavy ML deps before any app import ──────────────────────────────
_torch = types.ModuleType("torch")
_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", _torch)

_tf = types.ModuleType("transformers")
_tf.BioGptForCausalLM = object
_tf.BioGptTokenizer = object
_tf.pipeline = lambda *a, **k: None
sys.modules.setdefault("transformers", _tf)

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

    monkeypatch.setattr(report_generator, "load", lambda: None)
    monkeypatch.setattr(report_generator, "is_ready", lambda: True)
    monkeypatch.setattr(report_generator, "generate", _fake_generate)

    from app.main import app

    with TestClient(app) as c:
        yield c
