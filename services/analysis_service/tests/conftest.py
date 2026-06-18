"""
Test fixtures for analysis_service.

Runs the app on a throwaway SQLite DB, overrides the Redis-session auth, and
stubs Celery dispatch so the analyze endpoint never touches a real broker.
"""

import asyncio
import contextlib
import os
import sys
import tempfile
from collections.abc import Iterator
from unittest.mock import MagicMock

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="analysis_test_")
os.close(_DB_FD)
os.environ.update(
    DATABASE_URL=f"sqlite+aiosqlite:///{_DB_PATH}",
    REDIS_URL="redis://localhost:6379/0",
    CELERY_BROKER_URL="redis://localhost:6379/1",
    CELERY_RESULT_BACKEND="redis://localhost:6379/2",
    TRITON_HTTP_URL="triton:8000",
    GRADCAM_SERVICE_URL="http://gradcam_service:8004",
    ENVIRONMENT="test",
)

import pytest  # noqa: E402
from app.core.database import Base, engine  # noqa: E402
from app.core.deps import get_current_user_id  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TEST_USER_ID = 1


async def _drop_all() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def pytest_sessionfinish() -> None:
    with contextlib.suppress(OSError):
        os.remove(_DB_PATH)


@pytest.fixture(autouse=True)
def patch_celery(monkeypatch) -> MagicMock:
    """Replace Celery dispatch with a no-op mock (no broker in tests)."""
    mock = MagicMock()
    monkeypatch.setattr("app.workers.tasks.run_analysis.apply_async", mock)
    return mock


@pytest.fixture
def client() -> Iterator[TestClient]:
    asyncio.run(_drop_all())
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def anon_client() -> Iterator[TestClient]:
    asyncio.run(_drop_all())
    app.dependency_overrides.pop(get_current_user_id, None)
    with TestClient(app) as c:
        yield c
