"""API tests for the analyze / results endpoints."""

ANALYZE = "/api/v1/analyze/"
RESULTS = "/api/v1/results/"
# A realistic UUID (with hex letters) — an all-digits UUID would be coerced to
# a float by SQLite's NUMERIC affinity on the UUID column (test-only quirk).
STUDY_ID = "a1b2c3d4-e5f6-4789-abcd-ef0123456789"


def test_submit_analysis_queues_job(client, patch_celery):
    res = client.post(ANALYZE, json={"study_id": STUDY_ID, "task": "detection"})
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["status"] == "queued"
    assert body["job_id"]

    # the celery task was dispatched exactly once, keyed by job_id
    patch_celery.assert_called_once()
    _, kwargs = patch_celery.call_args
    assert kwargs["args"] == [body["job_id"]]
    assert kwargs["task_id"] == body["job_id"]


def test_submit_analysis_rejects_invalid_task(client):
    res = client.post(ANALYZE, json={"study_id": STUDY_ID, "task": "nope"})
    assert res.status_code == 422


def test_results_roundtrip(client):
    job_id = client.post(
        ANALYZE, json={"study_id": STUDY_ID, "task": "segmentation"}
    ).json()["job_id"]

    res = client.get(f"{RESULTS}{job_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"] == job_id
    assert body["task"] == "segmentation"
    assert body["status"] == "queued"
    assert body["results"] is None


def test_results_unknown_job_404(client):
    assert client.get(f"{RESULTS}00000000-0000-0000-0000-000000000000").status_code == 404


def test_endpoints_require_auth(anon_client):
    assert anon_client.post(ANALYZE, json={"study_id": STUDY_ID, "task": "detection"}).status_code == 401
    assert anon_client.get(f"{RESULTS}whatever").status_code == 401
