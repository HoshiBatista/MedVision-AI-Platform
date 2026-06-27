import structlog
from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.database import AsyncSessionFactory, create_tables
from app.core.logging_config import configure_logging
from app.services.report_generator import report_generator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text

configure_logging("report_service")
logger = structlog.get_logger()

app = FastAPI(
    title="MedVision Report Service",
    version="1.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(api_v1_router, prefix="/api/v1")


@app.on_event("startup")
async def startup() -> None:
    if settings.auto_create_tables:
        await create_tables()
    # Pull the model into Ollama in the background so the HTTP server is available
    # immediately. /ready returns model=false until the pull finishes.
    import asyncio

    asyncio.create_task(report_generator.ensure_model())

    logger.info(
        "report_service started",
        model=settings.llm_model_name,
        ollama_url=settings.ollama_url,
        environment=settings.environment,
    )


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready() -> dict:
    checks: dict[str, bool] = {}

    try:
        async with AsyncSessionFactory() as session:
            await session.execute(text("SELECT 1"))
        checks["db"] = True
    except Exception:
        checks["db"] = False

    checks["model"] = report_generator.is_ready()

    all_ok = all(checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
