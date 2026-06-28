import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.metrics import AUTH_PASSWORD_HASH_DURATION

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    with AUTH_PASSWORD_HASH_DURATION.time():
        return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    with AUTH_PASSWORD_HASH_DURATION.time():
        return pwd_context.verify(plain, hashed)


def create_access_token(user_id: int, role: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict = {"sub": str(user_id), "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def create_refresh_token() -> str:
    """Generate a fresh opaque refresh token (only its hash is stored)."""
    return secrets.token_urlsafe(48)


def hash_token(raw: str) -> str:
    """SHA-256 hex digest used to look up / store refresh tokens (never the raw value)."""
    return hashlib.sha256(raw.encode()).hexdigest()
