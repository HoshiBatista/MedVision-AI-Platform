from app.api.v1.endpoints import reports
from fastapi import APIRouter

router = APIRouter()
router.include_router(reports.router, prefix="/reports", tags=["reports"])
