from collections.abc import AsyncGenerator

from app.core.database import AsyncSessionFactory
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session
