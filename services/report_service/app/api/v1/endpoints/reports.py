import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.prompt_builder import build_prompt
from app.models.report import Report
from app.schemas.report import GenerateReportRequest, ReportResponse
from app.services.report_generator import report_generator

router = APIRouter()
logger = structlog.get_logger()


async def _run_generation(report_id: str, prompt: str) -> None:
    from app.core.database import AsyncSessionFactory

    async with AsyncSessionFactory() as db:
        result = await db.execute(select(Report).where(Report.id == report_id))
        report = result.scalar_one_or_none()
        if report is None:
            return

        report.status = "generating"
        await db.commit()

        try:
            content = await report_generator.generate(prompt)
            report.content = content
            report.status = "completed"
            logger.info("report generated", report_id=report_id, chars=len(content))
        except Exception as exc:
            report.status = "failed"
            report.error = str(exc)
            logger.error("report generation failed", report_id=report_id, error=str(exc))

        await db.commit()


@router.post("/generate", response_model=ReportResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_report(
    body: GenerateReportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Report:
    if body.modality.upper() not in ("MRI", "CXR", "DERM"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="modality must be MRI, CXR, or DERM",
        )

    if not report_generator.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet, retry in a moment",
        )

    prompt = build_prompt(body.modality, body.findings, body.patient_context)

    report = Report(
        study_id=body.study_id,
        user_id=0,  # auth integration point — pass user_id from JWT when gateway forwards it
        modality=body.modality.upper(),
        status="pending",
        findings_snapshot=body.findings.model_dump(),
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    background_tasks.add_task(_run_generation, report.id, prompt)

    logger.info("report queued", report_id=report.id, study_id=body.study_id, modality=body.modality)
    return report


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str, db: AsyncSession = Depends(get_db)) -> Report:
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.get("/study/{study_id}", response_model=list[ReportResponse])
async def list_reports_for_study(study_id: str, db: AsyncSession = Depends(get_db)) -> list[Report]:
    result = await db.execute(select(Report).where(Report.study_id == study_id))
    return list(result.scalars().all())
