import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.gradcam import compute_cam
from app.services.heatmap_renderer import render_and_save

router = APIRouter()
logger = structlog.get_logger()

_VALID_MODELS = {"skin_classification", "pneumonia_detection", "mri_segmentation"}


class ExplainRequest(BaseModel):
    model_name: str
    image_path: str
    target_class: int | None = None


class ExplainResponse(BaseModel):
    heatmap_path: str
    cam_shape: list[int]
    top_regions: list[dict]
    num_detections: int


@router.post("/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest) -> ExplainResponse:
    if body.model_name not in _VALID_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"model_name must be one of {sorted(_VALID_MODELS)}",
        )

    try:
        cam, meta = compute_cam(
            model_name=body.model_name,
            image_path=body.image_path,
            target_class=body.target_class,
        )
    except FileNotFoundError as exc:
        logger.warning("model file not found", error=str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("EigenCAM failed", model=body.model_name, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CAM computation failed: {exc}",
        )

    try:
        heatmap_path = render_and_save(
            cam=cam,
            image_path=body.image_path,
            model_name=body.model_name,
            target_class=body.target_class,
        )
    except Exception as exc:
        logger.error("heatmap rendering failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heatmap rendering failed: {exc}",
        )

    logger.info(
        "explain complete",
        model=body.model_name,
        heatmap_path=heatmap_path,
        target_class=body.target_class,
    )
    return ExplainResponse(
        heatmap_path=heatmap_path,
        cam_shape=meta["cam_shape"],
        top_regions=meta["top_regions"],
        num_detections=meta["num_detections"],
    )
