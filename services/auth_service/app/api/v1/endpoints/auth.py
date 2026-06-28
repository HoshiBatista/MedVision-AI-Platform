import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user, get_db
from app.core.metrics import (
    AUTH_LOGIN_TOTAL,
    AUTH_LOGOUT_TOTAL,
    AUTH_REFRESH_TOTAL,
    AUTH_REGISTER_TOTAL,
)
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.user import RefreshRequest, RegisterRequest, TokenResponse, UserResponse
from app.services.token_service import (
    RefreshError,
    issue_refresh_token,
    revoke_all_for_user,
    rotate_refresh_token,
)

router = APIRouter()
logger = structlog.get_logger()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)) -> User:
    logger.info("registration attempt", email=body.email)

    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        AUTH_REGISTER_TOTAL.labels(result="email_exists").inc()
        logger.warning("registration failed — email already exists", email=body.email)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    AUTH_REGISTER_TOTAL.labels(result="success").inc()
    logger.info("user registered", user_id=user.id, email=user.email, role=user.role)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    logger.info("login attempt", username=form.username)

    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(form.password, user.hashed_password):
        AUTH_LOGIN_TOTAL.labels(result="invalid_credentials").inc()
        logger.warning("login failed — invalid credentials", username=form.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        AUTH_LOGIN_TOTAL.labels(result="account_disabled").inc()
        logger.warning("login failed — account disabled", user_id=user.id)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    access_token = create_access_token(user.id, user.role)
    refresh_token = await issue_refresh_token(db, user.id)

    AUTH_LOGIN_TOTAL.labels(result="success").inc()
    logger.info("user logged in", user_id=user.id, email=user.email, role=user.role)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id, new_refresh = await rotate_refresh_token(db, body.refresh_token)
    except RefreshError as exc:
        AUTH_REFRESH_TOTAL.labels(result=exc.result).inc()
        logger.warning("refresh rejected", reason=str(exc))
        raise invalid from exc

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        # Token was valid but the account is gone/disabled — kill the session.
        await revoke_all_for_user(db, user_id)
        AUTH_REFRESH_TOTAL.labels(result="inactive").inc()
        raise invalid

    access_token = create_access_token(user.id, user.role)
    AUTH_REFRESH_TOTAL.labels(result="success").inc()
    logger.info("token refreshed", user_id=user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    revoked = await revoke_all_for_user(db, current_user.id)
    AUTH_LOGOUT_TOTAL.inc()
    logger.info(
        "user logged out",
        user_id=current_user.id,
        email=current_user.email,
        revoked_tokens=revoked,
    )
