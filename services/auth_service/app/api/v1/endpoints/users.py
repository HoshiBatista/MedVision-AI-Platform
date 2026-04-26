import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.user import UpdateUserRequest, UserResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> User:
    logger.debug("profile fetched", user_id=current_user.id, email=current_user.email)
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UpdateUserRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    changed_fields = []
    if body.full_name is not None:
        current_user.full_name = body.full_name
        changed_fields.append("full_name")

    await db.commit()
    await db.refresh(current_user)

    logger.info("profile updated", user_id=current_user.id, changed_fields=changed_fields)
    return current_user
