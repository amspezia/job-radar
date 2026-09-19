from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class EvalsBase(DeclarativeBase):
    pass


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


@lru_cache
def get_engine() -> AsyncEngine:
    # Imported here so that importing this module never requires EVALS_DATABASE_URL
    # (tests point at a throwaway database instead).
    from eval.evals_db.settings import get_settings

    return make_engine(get_settings().evals_database_url)


@asynccontextmanager
async def evals_session(url: str | None = None) -> AsyncIterator[AsyncSession]:
    """Yield a session on the EVALS database: the configured one, or `url` when given."""
    engine = make_engine(url) if url else get_engine()
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        if url:
            await engine.dispose()
