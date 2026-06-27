"""
Celery tasks for the analysis pipeline.

Flow per job:
  1. Mark job as 'running'
  2. Fetch study file path from DB
  3. Run Triton inference
  4. Request GradCAM heatmap (non-fatal on failure)
  5. Persist results + mark job 'completed'
  On any exception → mark job 'failed' with error message
"""

import time
from typing import Any

import structlog
from celery import Task
from sqlalchemy import text

from app.core.database import SyncSessionFactory
from app.core.metrics import ANALYSIS_DURATION_SECONDS, ANALYSIS_JOBS_TOTAL
from app.models.job import AnalysisJob
from app.services.gradcam_service import request_heatmap
from app.services.inference import get_inference_client
from app.services.triton_client import _TASK_TO_MODEL
from app.workers.celery_app import celery

logger = structlog.get_logger()


@celery.task(
    bind=True,
    name="analysis.run",
    max_retries=2,
    default_retry_delay=10,
)
def run_analysis(self: Task, job_id: str) -> dict[str, Any]:
    structlog.contextvars.bind_contextvars(job_id=job_id)
    start = time.perf_counter()

    with SyncSessionFactory() as session:
        job: AnalysisJob | None = session.get(AnalysisJob, job_id)
        if job is None:
            logger.error("job not found", job_id=job_id)
            return {"status": "not_found"}

        task = job.task
        study_id = job.study_id

        job.status = "running"
        session.commit()
        logger.info("job started", task=task, study_id=study_id)

        try:
            row = session.execute(
                text("SELECT file_path FROM studies WHERE id = :sid"),
                {"sid": study_id},
            ).fetchone()

            if row is None:
                raise ValueError(f"Study {study_id} not found in DB")

            file_path: str = row[0]
            logger.info("study file located", file_path=file_path)

            client = get_inference_client()
            infer_result = client.run(task=task, file_path=file_path)

            model_name = _TASK_TO_MODEL[task]
            top_class = None
            if infer_result["findings"]:
                top_class = infer_result["findings"][0].get("class_id")

            heatmap_path = request_heatmap(
                model_name=model_name,
                image_path=file_path,
                target_class=top_class,
            )

            elapsed = time.perf_counter() - start
            results: dict[str, Any] = {
                **infer_result,
                "heatmap_path": heatmap_path,
                "processing_time_s": round(elapsed, 3),
            }

            job.status = "completed"
            job.results = results
            session.commit()

            ANALYSIS_DURATION_SECONDS.labels(task=task).observe(elapsed)
            ANALYSIS_JOBS_TOTAL.labels(task=task, result="completed").inc()

            logger.info(
                "job completed",
                task=task,
                num_findings=infer_result["num_detections"],
                duration_s=round(elapsed, 2),
            )
            return {"status": "completed", "job_id": job_id}

        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_msg = f"{type(exc).__name__}: {exc}"

            job.status = "failed"
            job.error = error_msg
            session.commit()

            ANALYSIS_JOBS_TOTAL.labels(task=task, result="failed").inc()
            logger.error("job failed", task=task, error=error_msg, duration_s=round(elapsed, 2))

            try:
                raise self.retry(exc=exc)
            except self.MaxRetriesExceededError:
                return {"status": "failed", "job_id": job_id, "error": error_msg}
