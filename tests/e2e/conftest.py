"""
Shared fixtures for the docker-compose e2e suite.

These tests run against a LIVE stack (see `make e2e` / the CI `e2e` job), talking
to the Nginx gateway. They do not import service code; everything goes over HTTP.

Config via env:
  E2E_BASE_URL       gateway base URL (default http://localhost)
  E2E_READY_TIMEOUT  seconds to wait for the stack to become healthy (default 180)
  E2E_DIRECT_HOST    host where service ports are published (default localhost)
"""

import base64
import os
import time
import uuid

import httpx
import pytest

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost").rstrip("/")
DIRECT_HOST = os.environ.get("E2E_DIRECT_HOST", "localhost")
READY_TIMEOUT = float(os.environ.get("E2E_READY_TIMEOUT", "180"))

# Service ports published by docker-compose (for per-service /health checks).
SERVICE_PORTS = {
    "auth_service": 8001,
    "upload_service": 8002,
    "analysis_service": 8003,
    "gradcam_service": 8004,
    "report_service": 8005,
}

# A valid, Pillow-decodable 1x1 PNG.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _wait_until_ready() -> None:
    """Block until the gateway answers /health, or fail the session."""
    deadline = time.time() + READY_TIMEOUT
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(3)
    pytest.fail(f"stack not ready at {BASE_URL} within {READY_TIMEOUT}s (last error: {last_err})")


@pytest.fixture(scope="session", autouse=True)
def stack_ready() -> None:
    _wait_until_ready()


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30, follow_redirects=True) as c:
        yield c


@pytest.fixture(scope="session")
def user_credentials() -> dict[str, str]:
    # Unique per run so repeated e2e runs don't collide on the users table.
    return {
        "email": f"e2e+{uuid.uuid4().hex[:12]}@example.com",
        "password": "E2ePassw0rd!",
        "full_name": "E2E Tester",
    }


@pytest.fixture(scope="session")
def auth_token(client: httpx.Client, user_credentials: dict[str, str]) -> str:
    # Register (idempotent-ish: 409 if it already exists), then log in.
    reg = client.post("/api/v1/auth/register", json=user_credentials)
    assert reg.status_code in (201, 409), reg.text

    login = client.post(
        "/api/v1/auth/login",
        data={"username": user_credentials["email"], "password": user_credentials["password"]},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    assert token
    return token


@pytest.fixture(scope="session")
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}
