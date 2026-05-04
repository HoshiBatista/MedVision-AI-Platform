from fastapi import APIRouter

from app.api.v1.endpoints import reports

router = APIRouter()
router.include_router(reports.router, prefix="/reports", tags=["reports"])
