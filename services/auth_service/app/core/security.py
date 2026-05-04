from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
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
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict = {"sub": str(user_id), "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
