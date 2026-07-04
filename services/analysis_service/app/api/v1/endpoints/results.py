import structlog
from app.core.deps import get_current_user_id, get_db
from app.models.job import AnalysisJob
from app.schemas.job import JobListResponse, JobResultResponse
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

router = APIRouter()


@router.get("", response_model=JobListResponse)
async def list_results(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    study_id: str | None = Query(None, description="Filter jobs for a study"),
) -> JobListResponse:
    filters = [AnalysisJob.user_id == user_id]
    if study_id is not None:
        filters.append(AnalysisJob.study_id == study_id)

    total_result = await db.execute(
        select(func.count()).select_from(AnalysisJob).where(*filters)
    )
    total = total_result.scalar_one()

    result = await db.execute(
        select(AnalysisJob)
        .where(*filters)
        .order_by(AnalysisJob.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = list(result.scalars().all())

    logger.debug(
        "jobs listed",
        user_id=user_id,
        total=total,
        returned=len(items),
        offset=offset,
        study_id=study_id,
    )
    return JobListResponse(items=items, total=total)


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
