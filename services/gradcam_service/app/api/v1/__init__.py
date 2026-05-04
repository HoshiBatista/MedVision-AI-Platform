from fastapi import APIRouter

from app.api.v1.endpoints import explain

router = APIRouter()
router.include_router(explain.router, prefix="", tags=["explain"])
