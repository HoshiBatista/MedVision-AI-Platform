"""
Unit tests for the Celery analysis worker (`app.workers.tasks.run_analysis`).

Triton inference and the GradCAM HTTP call are mocked; the database is a real
SQLite file (the same one the async app uses, via SyncSessionFactory). The task
is executed in-process with `.apply()` so no broker/worker is required.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from app.core.database import Base, SyncSessionFactory, sync_engine
from app.models.job import AnalysisJob
from app.workers.tasks import run_analysis
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import text

# UUID with hex letters — an all-digit UUID would hit SQLite NUMERIC affinity on
# the UUID columns (see test_api.py).
STUDY_ID = "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
FILE_PATH = "/data/studies/a1b2c3d4/original.png"


@pytest.fixture(autouse=True)
def fresh_db():
    """Recreate analysis_jobs (from ORM metadata) and a minimal studies table."""
    Base.metadata.drop_all(sync_engine)
    Base.metadata.create_all(sync_engine)
    with sync_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS studies"))
        conn.execute(text("CREATE TABLE studies (id VARCHAR PRIMARY KEY, file_path VARCHAR)"))
    yield
    Base.metadata.drop_all(sync_engine)
    with sync_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS studies"))


@pytest.fixture
def seed_job():
    """Insert an optional study row + a queued job; return the job id."""

    def _seed(task: str, *, with_study: bool = True, file_path: str = FILE_PATH) -> str:
        job_id = str(uuid.uuid4())
        with SyncSessionFactory() as session:
            if with_study:
                session.execute(
                    text("INSERT INTO studies (id, file_path) VALUES (:i, :p)"),
                    {"i": STUDY_ID, "p": file_path},
                )
            session.add(
                AnalysisJob(
                    id=job_id,
                    study_id=STUDY_ID,
                    user_id=1,
                    task=task,
                    status="queued",
                )
            )
            session.commit()
        return job_id

    return _seed


@pytest.fixture
def no_retry(monkeypatch):
    """Make self.retry() terminal so the failure branch returns immediately."""

    def _raise(*_args, **_kwargs):
        raise MaxRetriesExceededError()

    monkeypatch.setattr(run_analysis, "retry", _raise)


def _read_job(job_id: str) -> AnalysisJob:
    with SyncSessionFactory() as session:
        job = session.get(AnalysisJob, job_id)
        assert job is not None
        session.expunge(job)
        return job


def _mock_triton(monkeypatch, infer_result: dict) -> MagicMock:
    client = MagicMock()
    client.run = MagicMock(return_value=infer_result)
    monkeypatch.setattr("app.workers.tasks.get_inference_client", lambda: client)
    return client.run


# ─────────────────────────────── happy path ────────────────────────────────
def test_run_analysis_completes_and_persists_results(monkeypatch, seed_job):
    job_id = seed_job("detection")
    run = _mock_triton(
        monkeypatch,
        {"findings": [{"class_id": 2, "confidence": 0.91}], "num_detections": 1, "confidence": 0.91},
    )
    heat = MagicMock(return_value="/data/heatmaps/a1b2c3d4.png")
    monkeypatch.setattr("app.workers.tasks.request_heatmap", heat)

    out = run_analysis.apply(args=[job_id]).result

    assert out == {"status": "completed", "job_id": job_id}

    # Triton called with the resolved task + study file path
    run.assert_called_once_with(task="detection", file_path=FILE_PATH)

    # GradCAM called with the mapped model name + top finding's class
    heat.assert_called_once_with(
        model_name="pneumonia_detection", image_path=FILE_PATH, target_class=2
    )

    job = _read_job(job_id)
    assert job.status == "completed"
    assert job.error is None
    assert job.results["heatmap_path"] == "/data/heatmaps/a1b2c3d4.png"
    assert job.results["num_detections"] == 1
    assert "processing_time_s" in job.results


def test_run_analysis_maps_task_to_segmentation_model(monkeypatch, seed_job):
    job_id = seed_job("segmentation")
    _mock_triton(monkeypatch, {"findings": [{"class_id": 0}], "num_detections": 1})
    heat = MagicMock(return_value=None)
    monkeypatch.setattr("app.workers.tasks.request_heatmap", heat)

    run_analysis.apply(args=[job_id])

    assert heat.call_args.kwargs["model_name"] == "mri_segmentation"


def test_run_analysis_no_findings_passes_none_target(monkeypatch, seed_job):
    job_id = seed_job("classification")
    _mock_triton(monkeypatch, {"findings": [], "num_detections": 0})
    heat = MagicMock(return_value=None)
    monkeypatch.setattr("app.workers.tasks.request_heatmap", heat)

    out = run_analysis.apply(args=[job_id]).result

    assert out["status"] == "completed"
    assert heat.call_args.kwargs["target_class"] is None
    assert heat.call_args.kwargs["model_name"] == "skin_classification"


# ─────────────────────────── gradcam is non-fatal ──────────────────────────
def test_run_analysis_gradcam_none_still_completes(monkeypatch, seed_job):
    job_id = seed_job("detection")
    _mock_triton(monkeypatch, {"findings": [{"class_id": 1}], "num_detections": 1})
    monkeypatch.setattr("app.workers.tasks.request_heatmap", MagicMock(return_value=None))

    out = run_analysis.apply(args=[job_id]).result

    assert out["status"] == "completed"
    job = _read_job(job_id)
    assert job.status == "completed"
    assert job.results["heatmap_path"] is None


# ───────────────────────────── failure paths ───────────────────────────────
def test_run_analysis_missing_job_returns_not_found(monkeypatch):
    # No DB row for this id; task short-circuits before touching inference.
    triton = MagicMock()
    monkeypatch.setattr("app.workers.tasks.get_inference_client", lambda: triton)

    out = run_analysis.apply(args=["deadbeef-0000-4000-8000-000000000abc"]).result

    assert out == {"status": "not_found"}
    triton.run.assert_not_called()


def test_run_analysis_missing_study_marks_failed(monkeypatch, seed_job, no_retry):
    job_id = seed_job("detection", with_study=False)
    run = _mock_triton(monkeypatch, {"findings": [], "num_detections": 0})

    out = run_analysis.apply(args=[job_id]).result

    assert out["status"] == "failed"
    assert out["job_id"] == job_id
    assert "not found in DB" in out["error"]
    run.assert_not_called()  # failed during study lookup, before inference

    job = _read_job(job_id)
    assert job.status == "failed"
    assert job.error and "not found in DB" in job.error


def test_run_analysis_triton_error_marks_failed(monkeypatch, seed_job, no_retry):
    job_id = seed_job("detection")
    client = MagicMock()
    client.run = MagicMock(side_effect=RuntimeError("triton unreachable"))
    monkeypatch.setattr("app.workers.tasks.get_inference_client", lambda: client)
    heat = MagicMock()
    monkeypatch.setattr("app.workers.tasks.request_heatmap", heat)

    out = run_analysis.apply(args=[job_id]).result

    assert out["status"] == "failed"
    assert "triton unreachable" in out["error"]
    heat.assert_not_called()  # inference failed before the heatmap step

    job = _read_job(job_id)
    assert job.status == "failed"
    assert "RuntimeError" in job.error
