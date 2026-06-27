"""
End-to-end smoke + contract tests against a live docker-compose stack.

Covers the wiring that per-service unit tests cannot: JWT issued by auth_service
is accepted by upload/analysis, the gateway routes correctly, and the
upload -> analyze -> results pipeline runs a real prediction. The default stack
uses the local ONNX Runtime (CPU) inference backend, so the analysis job is
expected to reach 'completed' with structured inference output — no GPU/Triton
needed. (BioGPT report generation is still not exercised here.)
"""

import time

import httpx
import pytest
from conftest import DIRECT_HOST, PNG_1X1, SERVICE_PORTS

TERMINAL = {"completed", "failed"}
RESULT_TIMEOUT = 120


def test_gateway_health(client: httpx.Client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    assert r.json().get("status") == "ok"


@pytest.mark.parametrize("service", sorted(SERVICE_PORTS))
def test_service_health_direct(service: str):
    port = SERVICE_PORTS[service]
    r = httpx.get(f"http://{DIRECT_HOST}:{port}/health", timeout=10)
    assert r.status_code == 200, f"{service}: {r.text}"
    assert r.json().get("status") == "ok"


def test_login_and_me(client: httpx.Client, auth_headers: dict, user_credentials: dict):
    r = client.get("/api/v1/users/me", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["email"] == user_credentials["email"]


def test_upload_requires_auth(client: httpx.Client):
    # No Bearer token -> upload_service must reject (proves JWT auth is enforced).
    r = client.post(
        "/api/v1/upload?modality=DERM",
        files={"file": ("x.png", PNG_1X1, "image/png")},
    )
    assert r.status_code == 401, r.text


def test_pipeline_upload_analyze_results(client: httpx.Client, auth_headers: dict):
    # 1. Upload through the gateway with the JWT (proves the gateway path fix +
    #    cross-service JWT auth: auth_service token accepted by upload_service).
    up = client.post(
        "/api/v1/upload?modality=DERM",
        files={"file": ("lesion.png", PNG_1X1, "image/png")},
        headers=auth_headers,
    )
    assert up.status_code == 201, up.text
    study_id = up.json()["id"]
    assert study_id

    # 2. Submit analysis (analysis_service must also accept the same JWT).
    an = client.post(
        "/api/v1/analyze/",
        json={"study_id": study_id, "task": "classification"},
        headers=auth_headers,
    )
    assert an.status_code == 202, an.text
    job_id = an.json()["job_id"]
    assert job_id

    # 3. Poll until the worker drives the job to a terminal state. The local ONNX
    #    backend runs a real prediction on CPU, so we expect 'completed'.
    deadline = time.time() + RESULT_TIMEOUT
    body: dict = {}
    while time.time() < deadline:
        res = client.get(f"/api/v1/results/{job_id}", headers=auth_headers)
        assert res.status_code == 200, res.text
        body = res.json()
        if body["status"] in TERMINAL:
            break
        time.sleep(3)

    assert body.get("status") in TERMINAL, f"job did not finish in {RESULT_TIMEOUT}s: {body}"
    assert body["job_id"] == job_id
    assert body["task"] == "classification"

    # Real inference must have succeeded (not just reached a terminal state).
    assert body["status"] == "completed", f"analysis failed: {body.get('error')}"
    results = body.get("results") or {}
    assert results.get("model") == "skin_classification", results
    assert "num_detections" in results, results
    assert results.get("orig_size") == [1, 1], results  # 1x1 PNG fixture echoed back
