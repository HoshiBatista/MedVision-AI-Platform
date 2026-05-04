import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user, get_db
from app.core.metrics import AUTH_LOGIN_TOTAL, AUTH_LOGOUT_TOTAL, AUTH_REGISTER_TOTAL
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.user import RegisterRequest, TokenResponse, UserResponse

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

    AUTH_LOGIN_TOTAL.labels(result="success").inc()
    logger.info("user logged in", user_id=user.id, email=user.email, role=user.role)

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: User = Depends(get_current_user)) -> None:
    AUTH_LOGOUT_TOTAL.inc()
    logger.info("user logged out", user_id=current_user.id, email=current_user.email)
