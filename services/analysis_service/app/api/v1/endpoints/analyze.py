import uuid

import structlog
from app.core.deps import get_current_user_id, get_db
from app.core.metrics import ANALYSIS_JOBS_TOTAL
from app.models.job import VALID_TASKS, AnalysisJob
from app.schemas.job import AnalyzeRequest, JobQueuedResponse
from app.workers.tasks import run_analysis
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

router = APIRouter()


@router.post("/", response_model=JobQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_analysis(
    body: AnalyzeRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> JobQueuedResponse:
    if body.task not in VALID_TASKS:
        raise HTTPException(status_code=422, detail=f"task must be one of {VALID_TASKS}")

    job_id = str(uuid.uuid4())
    job = AnalysisJob(
        id=job_id,
        study_id=body.study_id,
        user_id=user_id,
        task=body.task,
        status="queued",
    )
    db.add(job)
    await db.commit()

    run_analysis.apply_async(args=[job_id], task_id=job_id)

    ANALYSIS_JOBS_TOTAL.labels(task=body.task, result="queued").inc()
    logger.info("job queued", job_id=job_id, study_id=body.study_id, task=body.task, user_id=user_id)

    return JobQueuedResponse(job_id=job_id, status="queued")
