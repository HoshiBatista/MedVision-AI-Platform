"""
Test fixtures for upload_service.

Runs the app on a throwaway SQLite DB and a temp storage root, and overrides
the JWT auth dependency so endpoints can be exercised without a real token.
"""

import asyncio
import contextlib
import io
import os
import sys
import tempfile
from collections.abc import Iterator

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

_STORAGE_ROOT = tempfile.mkdtemp(prefix="upload_store_")
_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="upload_test_")
os.close(_DB_FD)
os.environ.update(
    DATABASE_URL=f"sqlite+aiosqlite:///{_DB_PATH}",
    REDIS_URL="redis://localhost:6379/0",
    STORAGE_ROOT=_STORAGE_ROOT,
    ENVIRONMENT="test",
)

import pytest  # noqa: E402
from app.core.database import Base, engine  # noqa: E402
from app.core.deps import get_current_user_id  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from pydicom.data import get_testdata_file  # noqa: E402

TEST_USER_ID = 1


async def _drop_all() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def pytest_sessionfinish() -> None:
    with contextlib.suppress(OSError):
        os.remove(_DB_PATH)


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


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def dicom_bytes() -> bytes:
    # CT_small.dcm ships with pydicom (no network); has PixelData + rescale tags.
    with open(get_testdata_file("CT_small.dcm"), "rb") as f:
        return f.read()
