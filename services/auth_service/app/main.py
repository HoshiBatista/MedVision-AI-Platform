import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import select, text

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.database import AsyncSessionFactory, create_tables
from app.core.logging_config import configure_logging
from app.core.security import hash_password
from app.middleware.logging import RequestLoggingMiddleware
from app.models.user import User

configure_logging("auth_service")
logger = structlog.get_logger()

app = FastAPI(
    title="MedVision Auth Service",
    version="1.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url=None,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(api_v1_router, prefix="/api/v1")


async def _seed_admin() -> None:
    async with AsyncSessionFactory() as db:
        result = await db.execute(select(User).where(User.email == settings.admin_username))
        if result.scalar_one_or_none() is not None:
            return
        admin = User(
            email=settings.admin_username,
            hashed_password=hash_password(settings.admin_password),
            full_name="Administrator",
            role="admin",
        )
        db.add(admin)
        await db.commit()
        logger.info("admin user seeded", username=settings.admin_username)


@app.on_event("startup")
async def startup() -> None:
    await create_tables()
    await _seed_admin()
    logger.info(
        "auth_service started",
        environment=settings.environment,
        log_level=settings.log_level,
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

    all_ok = all(checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
