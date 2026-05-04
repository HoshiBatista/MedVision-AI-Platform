import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_admin_user, get_db
from app.models.user import User
from app.schemas.user import AdminUpdateUserRequest, UserResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> list[User]:
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = list(result.scalars().all())
    logger.info("admin listed users", admin_id=admin.id, count=len(users))
    return users


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    body: AdminUpdateUserRequest,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    changed_fields = []
    if body.full_name is not None:
        user.full_name = body.full_name
        changed_fields.append("full_name")
    if body.role is not None:
        user.role = body.role
        changed_fields.append("role")
    if body.is_active is not None:
        user.is_active = body.is_active
        changed_fields.append("is_active")

    await db.commit()
    await db.refresh(user)

    logger.info("admin updated user", admin_id=admin.id, target_user_id=user_id, changed_fields=changed_fields)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot deactivate yourself")

    user.is_active = False
    await db.commit()

    logger.info("admin deactivated user", admin_id=admin.id, target_user_id=user_id)
