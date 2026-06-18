"""Test fixtures for gradcam_service (no DB / no auth)."""

import os
import sys
import tempfile
from collections.abc import Iterator

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

os.environ.update(
    HEATMAP_OUTPUT_DIR=tempfile.mkdtemp(prefix="gradcam_heatmaps_"),
    MODEL_REPO_ROOT=tempfile.mkdtemp(prefix="gradcam_models_"),
    ENVIRONMENT="test",
)

import pytest  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
