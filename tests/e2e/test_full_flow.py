"""
End-to-end smoke + contract tests against a live docker-compose stack.

Covers the wiring that per-service unit tests cannot: JWT issued by auth_service
is accepted by upload/analysis, the gateway routes correctly, and the
full happy-path runs for real on CPU: upload -> prediction (local ONNX Runtime)
-> EigenCAM heatmap (served through the gateway) -> BioGPT clinical report. No
GPU/Triton needed.
"""

import time

import httpx
import pytest
from conftest import DIRECT_HOST, PNG_1X1, SERVICE_PORTS

TERMINAL = {"completed", "failed"}
RESULT_TIMEOUT = 120
# BioGPT loads lazily in the background at startup (the container is "healthy"
# before the model is ready), and a fresh CI run re-downloads the weights, so the
# readiness wait is generous; CPU generation of one report is then bounded too.
REPORT_READY_TIMEOUT = 600
REPORT_GEN_TIMEOUT = 240


def _wait_report_model_ready() -> None:
    """Block until report_service reports its BioGPT model is loaded."""
    url = f"http://{DIRECT_HOST}:{SERVICE_PORTS['report_service']}/ready"
    deadline = time.time() + REPORT_READY_TIMEOUT
    last: str | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=10)
            if r.status_code == 200 and r.json().get("checks", {}).get("model"):
                return
            last = r.text
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(5)
    pytest.fail(f"BioGPT not ready within {REPORT_READY_TIMEOUT}s (last: {last})")


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
    assert results.get("orig_size") == [8, 8], results  # 8x8 PNG fixture echoed back

    # The EigenCAM heatmap must have been generated (gradcam_service read the
    # uploaded study, computed the CAM, and wrote the PNG) and be retrievable
    # through the gateway's static route — the full predict -> explain path on CPU.
    heatmap_path = results.get("heatmap_path")
    assert heatmap_path and heatmap_path.startswith("/data/heatmaps/"), results
    hm = client.get(f"/static/heatmaps/{heatmap_path.rsplit('/', 1)[-1]}")
    assert hm.status_code == 200, hm.text
    assert hm.headers.get("content-type", "").startswith("image/"), dict(hm.headers)

    # 4. Generate a BioGPT clinical report from the findings (CPU). The model
    #    loads lazily, so wait for report_service to report it ready first.
    _wait_report_model_ready()
    gen = client.post(
        "/api/v1/reports/generate",
        json={
            "study_id": study_id,
            "modality": "DERM",
            "job_ids": [job_id],
            "findings": {
                "confidence": results.get("top_confidence"),
                "bbox_count": results.get("num_detections"),
                "labels": ["lesion"],
                "raw": {"model": results.get("model")},
            },
            "patient_context": {"clinical_indication": "e2e smoke test"},
        },
        headers=auth_headers,
    )
    assert gen.status_code == 202, gen.text
    report_id = gen.json()["report_id"]
    assert report_id

    # 5. Poll until BioGPT finishes generating the report.
    deadline = time.time() + REPORT_GEN_TIMEOUT
    rbody: dict = {}
    while time.time() < deadline:
        rr = client.get(f"/api/v1/reports/{report_id}", headers=auth_headers)
        assert rr.status_code == 200, rr.text
        rbody = rr.json()
        if rbody["status"] in TERMINAL:
            break
        time.sleep(3)

    assert rbody.get("status") == "completed", f"report generation failed: {rbody}"
    assert rbody.get("content"), rbody
    assert "DISCLAIMER" in rbody["content"], rbody["content"][:500]
