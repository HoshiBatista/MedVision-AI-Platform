from collections.abc import AsyncGenerator

from fastapi import Cookie, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionFactory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session


async def get_redis() -> AsyncGenerator[Redis, None]:
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


async def get_current_user_id(
    medvision_session: str | None = Cookie(default=None, alias=settings.session_cookie_name),
    redis: Redis = Depends(get_redis),
) -> int:
    if not medvision_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    value = await redis.get(f"session:{medvision_session}")
    if value is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid")

    return int(value)
