"""
Test fixtures for auth_service.

The service is pointed at a throwaway SQLite database *before* `app` is
imported, so the real Postgres engine is never constructed. Each test gets a
clean schema with a freshly seeded admin user.
"""

import asyncio
import contextlib
import os
import sys
import tempfile
from collections.abc import Callable, Iterator

# ── Make `app` importable and configure a SQLite test DB BEFORE importing it ──
_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="auth_test_")
os.close(_DB_FD)
os.environ.update(
    DATABASE_URL=f"sqlite+aiosqlite:///{_DB_PATH}",
    JWT_SECRET_KEY="test-secret-key-only-for-tests",
    JWT_ALGORITHM="HS256",
    ACCESS_TOKEN_EXPIRE_MINUTES="30",
    ADMIN_USERNAME="admin",
    ADMIN_PASSWORD="admin",
    ENVIRONMENT="test",
)

import pytest  # noqa: E402
from app.core.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


async def _drop_all() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def pytest_sessionfinish() -> None:
    with contextlib.suppress(OSError):
        os.remove(_DB_PATH)


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Clean slate, then let startup create tables + seed the admin user.
    asyncio.run(_drop_all())
    with TestClient(app) as c:
        yield c


def _login(client: TestClient, username: str, password: str) -> str:
    res = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    res.raise_for_status()
    return res.json()["access_token"]


@pytest.fixture
def bearer() -> Callable[[str], dict[str, str]]:
    return lambda token: {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_token(client: TestClient) -> str:
    return _login(client, "admin", "admin")


@pytest.fixture
def user_token(client: TestClient) -> str:
    client.post(
        "/api/v1/auth/register",
        json={"email": "user@example.com", "password": "password123", "full_name": "Test User"},
    )
    return _login(client, "user@example.com", "password123")
