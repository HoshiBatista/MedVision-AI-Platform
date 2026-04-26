from fastapi import APIRouter

from app.api.v1.endpoints import upload

router = APIRouter()
router.include_router(upload.router, prefix="/studies", tags=["studies"])
