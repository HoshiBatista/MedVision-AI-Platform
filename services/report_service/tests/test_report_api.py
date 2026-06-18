"""API tests for the reports endpoints (generator mocked in conftest)."""

# substring of the mocked generator output (see conftest.FAKE_CONTENT)
FAKE_MARKER = "representative finding"

GEN = "/api/v1/reports/generate"
STUDY_ID = "a1b2c3d4-e5f6-4789-abcd-ef0123456789"


def _payload(modality="MRI"):
    return {
        "study_id": STUDY_ID,
        "modality": modality,
        "job_ids": ["b2c3d4e5-f6a7-4890-bcde-f01234567890"],
        "findings": {"confidence": 0.9, "bbox_count": 1, "top_class": "tumor"},
    }


def test_generate_queues_and_completes(client):
    res = client.post(GEN, json=_payload("MRI"))
    assert res.status_code == 202, res.text
    body = res.json()
    report_id = body["report_id"]
    assert report_id
    assert body["modality"] == "MRI"

    # background task ran with the mocked generator → completed with content
    fetched = client.get(f"/api/v1/reports/{report_id}")
    assert fetched.status_code == 200
    fbody = fetched.json()
    assert fbody["status"] == "completed"
    assert FAKE_MARKER in fbody["content"]


def test_generate_rejects_invalid_modality(client):
    res = client.post(GEN, json=_payload("ULTRASOUND"))
    assert res.status_code == 422


def test_generate_503_when_model_not_ready(client, monkeypatch):
    from app.services.report_generator import report_generator

    monkeypatch.setattr(report_generator, "is_ready", lambda: False)
    res = client.post(GEN, json=_payload("CXR"))
    assert res.status_code == 503


def test_get_unknown_report_404(client):
    assert client.get("/api/v1/reports/00000000-0000-0000-0000-000000000000").status_code == 404


def test_list_reports_for_study(client):
    report_id = client.post(GEN, json=_payload("DERM")).json()["report_id"]
    res = client.get(f"/api/v1/reports/study/{STUDY_ID}")
    assert res.status_code == 200
    ids = [r["report_id"] for r in res.json()]
    assert report_id in ids
