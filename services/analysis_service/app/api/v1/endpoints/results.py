import structlog
from app.core.deps import get_current_user_id, get_db
from app.models.job import AnalysisJob
from app.schemas.job import JobResultResponse
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

router = APIRouter()


@router.get("/{job_id}", response_model=JobResultResponse)
async def get_result(
    job_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> AnalysisJob:
    result = await db.execute(
        select(AnalysisJob).where(AnalysisJob.id == job_id, AnalysisJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()

    if job is None:
        logger.warning("job not found", job_id=job_id, user_id=user_id)
        raise HTTPException(status_code=404, detail="Job not found")

    logger.debug("job fetched", job_id=job_id, status=job.status)
    return job
