import time
import uuid

from passlib.context import CryptContext
from redis.asyncio import Redis

from app.core.config import settings
from app.core.metrics import AUTH_PASSWORD_HASH_DURATION

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    with AUTH_PASSWORD_HASH_DURATION.time():
        return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    with AUTH_PASSWORD_HASH_DURATION.time():
        return pwd_context.verify(plain, hashed)


async def create_session(redis: Redis, user_id: int) -> str:
    session_id = str(uuid.uuid4())
    await redis.setex(
        f"session:{session_id}",
        settings.session_ttl_seconds,
        str(user_id),
    )
    return session_id


async def get_user_id_from_session(redis: Redis, session_id: str) -> int | None:
    value = await redis.get(f"session:{session_id}")
    if value is None:
        return None
    return int(value)


async def delete_session(redis: Redis, session_id: str) -> None:
    await redis.delete(f"session:{session_id}")
