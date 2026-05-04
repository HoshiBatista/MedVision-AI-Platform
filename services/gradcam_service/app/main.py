import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging("gradcam_service")
logger = structlog.get_logger()

app = FastAPI(
    title="MedVision GradCAM Service",
    version="1.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(api_v1_router, prefix="/api/v1")


@app.on_event("startup")
async def startup() -> None:
    from pathlib import Path
    Path(settings.heatmap_output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "gradcam_service started",
        model_repo=settings.model_repo_root,
        heatmap_dir=settings.heatmap_output_dir,
        environment=settings.environment,
    )


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready() -> dict:
    from pathlib import Path
    checks = {
        "model_repo": Path(settings.model_repo_root).exists(),
        "heatmap_dir": Path(settings.heatmap_output_dir).exists(),
    }
    all_ok = all(checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
