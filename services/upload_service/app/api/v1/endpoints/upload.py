import time
import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user_id, get_db
from app.core.dicom_processor import DicomValidationError, validate_and_extract, validate_image
from app.core.metrics import (
    DICOM_VALIDATION_TOTAL,
    UPLOAD_DURATION_SECONDS,
    UPLOAD_FILE_SIZE_BYTES,
    UPLOADS_TOTAL,
)
from app.models.study import Study
from app.schemas.study import StudyListResponse, StudyResponse
from app.services.storage_service import save_file

logger = structlog.get_logger()

ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}
MODALITIES = {"MRI", "CXR", "DERM"}

router = APIRouter()


@router.post("/", response_model=StudyResponse, status_code=status.HTTP_201_CREATED)
async def upload_study(
    file: UploadFile,
    modality: Annotated[str, Query(description="MRI | CXR | DERM")],
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> Study:
    start = time.perf_counter()
    filename = file.filename or "unknown"

    logger.info(
        "upload started",
        filename=filename,
        modality=modality,
        user_id=user_id,
        content_type=file.content_type,
    )

    if modality not in MODALITIES:
        UPLOADS_TOTAL.labels(modality=modality, result="unsupported_modality").inc()
        logger.warning("upload rejected — invalid modality", modality=modality, user_id=user_id)
        raise HTTPException(status_code=422, detail=f"modality must be one of {MODALITIES}")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        UPLOADS_TOTAL.labels(modality=modality, result="unsupported_format").inc()
        logger.warning("upload rejected — unsupported format", suffix=suffix, filename=filename)
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {suffix}")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = await file.read()
    file_size = len(data)

    logger.debug("file read", filename=filename, size_bytes=file_size, size_mb=round(file_size / 1_048_576, 2))

    if file_size > max_bytes:
        UPLOADS_TOTAL.labels(modality=modality, result="too_large").inc()
        logger.warning(
            "upload rejected — file too large",
            size_mb=round(file_size / 1_048_576, 2),
            max_mb=settings.max_upload_size_mb,
            filename=filename,
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {round(file_size / 1_048_576, 1)} MB (max {settings.max_upload_size_mb} MB)",
        )

    # Validate file content
    meta: dict = {}
    is_dicom = suffix in {".dcm", ".dicom"}
    try:
        if is_dicom:
            logger.debug("starting DICOM validation", filename=filename)
            meta = validate_and_extract(data)
            DICOM_VALIDATION_TOTAL.labels(result="success").inc()
            logger.info(
                "DICOM validated",
                filename=filename,
                rows=meta.get("rows"),
                columns=meta.get("columns"),
                modality_tag=meta.get("modality_tag"),
                bits_allocated=meta.get("bits_allocated"),
                pixel_range=f"{meta.get('pixel_min', 0):.1f}–{meta.get('pixel_max', 0):.1f}",
            )
        else:
            meta = validate_image(data)
            logger.info(
                "image validated",
                filename=filename,
                width=meta.get("width"),
                height=meta.get("height"),
                mode=meta.get("mode"),
            )
    except (DicomValidationError, ValueError) as exc:
        if is_dicom:
            DICOM_VALIDATION_TOTAL.labels(result="error").inc()
        UPLOADS_TOTAL.labels(modality=modality, result="validation_error").inc()
        logger.error(
            "file validation failed",
            filename=filename,
            modality=modality,
            error=str(exc),
            is_dicom=is_dicom,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Persist to disk
    study_id = str(uuid.uuid4())
    safe_filename = f"original{suffix}"

    logger.debug("saving file to storage", study_id=study_id, path=f"{study_id}/{safe_filename}")
    file_path = await save_file(study_id, safe_filename, data)

    study = Study(
        id=study_id,
        user_id=user_id,
        modality=modality,
        original_filename=filename,
        file_path=file_path,
        file_size_bytes=file_size,
        meta=meta,
    )
    db.add(study)
    await db.commit()
    await db.refresh(study)

    duration_s = time.perf_counter() - start
    UPLOADS_TOTAL.labels(modality=modality, result="success").inc()
    UPLOAD_FILE_SIZE_BYTES.labels(modality=modality).observe(file_size)
    UPLOAD_DURATION_SECONDS.labels(modality=modality).observe(duration_s)

    logger.info(
        "study uploaded successfully",
        study_id=study_id,
        modality=modality,
        user_id=user_id,
        filename=filename,
        size_mb=round(file_size / 1_048_576, 2),
        duration_ms=round(duration_s * 1000, 2),
        is_dicom=is_dicom,
    )
    return study


@router.get("/", response_model=StudyListResponse)
async def list_studies(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> StudyListResponse:
    total_result = await db.execute(
        select(func.count()).select_from(Study).where(Study.user_id == user_id)
    )
    total = total_result.scalar_one()

    result = await db.execute(
        select(Study)
        .where(Study.user_id == user_id)
        .order_by(Study.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = list(result.scalars().all())

    logger.debug("studies listed", user_id=user_id, total=total, returned=len(items), offset=offset)
    return StudyListResponse(items=items, total=total)


@router.get("/{study_id}", response_model=StudyResponse)
async def get_study(
    study_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> Study:
    result = await db.execute(
        select(Study).where(Study.id == study_id, Study.user_id == user_id)
    )
    study = result.scalar_one_or_none()

    if study is None:
        logger.warning("study not found", study_id=study_id, user_id=user_id)
        raise HTTPException(status_code=404, detail="Study not found")

    logger.debug("study fetched", study_id=study_id, modality=study.modality, user_id=user_id)
    return study
