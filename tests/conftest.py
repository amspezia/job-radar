import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.base import async_session_factory


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    async with async_session_factory() as session:
        yield session
