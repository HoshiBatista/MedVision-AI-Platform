from fastapi import APIRouter

from app.api.v1.endpoints import admin, auth, users

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(users.router, prefix="/users", tags=["users"])
router.include_router(admin.router, prefix="/admin", tags=["admin"])
