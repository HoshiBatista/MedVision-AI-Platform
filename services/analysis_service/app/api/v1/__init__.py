from fastapi import APIRouter

from app.api.v1.endpoints import analyze, results

router = APIRouter()
router.include_router(analyze.router, prefix="/analyze", tags=["analysis"])
router.include_router(results.router, prefix="/results", tags=["results"])
