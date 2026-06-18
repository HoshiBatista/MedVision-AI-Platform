"""
HTTP client that calls the standalone gradcam_service.
Called from Celery tasks; uses synchronous httpx.
Failures are non-fatal — analysis results are stored without a heatmap.
"""

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger()

_TIMEOUT = 30.0


def request_heatmap(model_name: str, image_path: str, target_class: int | None = None) -> str | None:
    """
    POST /api/v1/explain to gradcam_service.
    Returns the local heatmap file path written by the service, or None on failure.
    """
    payload: dict = {"model_name": model_name, "image_path": image_path}
    if target_class is not None:
        payload["target_class"] = target_class

    try:
        resp = httpx.post(
            f"{settings.gradcam_service_url}/api/v1/explain",
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        heatmap_path: str | None = data.get("heatmap_path")
        logger.info("gradcam heatmap received", model=model_name, heatmap_path=heatmap_path)
        return heatmap_path
    except Exception as exc:
        logger.warning("gradcam service unavailable — skipping heatmap", error=str(exc))
        return None
