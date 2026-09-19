"""The harness's only door to the production database, and it opens one way.

Every connection is made with `-c default_transaction_read_only=on`. Verified against the
live database: it rejects CREATE, INSERT, UPDATE, DELETE and TRUNCATE — and an ORM flush —
with `ReadOnlySqlTransaction`, and a reconnect re-applies it.

It is not a sandbox. `SET default_transaction_read_only = off` and `SET TRANSACTION READ
WRITE` both defeat it from inside the session (review R-m4), and `SET` used to survive pool
checkouts — which is why the pool is `NullPool` and why every session re-reads
`transaction_read_only` on entry instead of trusting the connect option. The guard stops
accidents, not a deliberate override; "prod untouched" is backed by the before/after row
counts this module also provides.
"""

import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from eval.evals_db.settings import get_settings

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")

PROD_TABLES = ("jobs", "profile", "eval_labels", "fit_judgments")


class ProdNotReadOnly(RuntimeError):
    """The connection to production is not read-only — refuse to hand it out."""


@asynccontextmanager
async def prod_session(url: str | None = None) -> AsyncIterator[AsyncSession]:
    """Yield a read-only session on production (or on `url`, which tests point elsewhere)."""
    target = url or get_settings().database_url
    if not target:
        raise RuntimeError(
            "DATABASE_URL is not set: the harness needs it to read production (read-only)."
        )
    engine = create_async_engine(
        target,
        poolclass=NullPool,
        connect_args={"options": "-c default_transaction_read_only=on"},
    )
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            mode = await session.scalar(text("SHOW transaction_read_only"))
            if mode != "on":
                raise ProdNotReadOnly(
                    f"transaction_read_only is {mode!r}, expected 'on' — refusing to read "
                    "production through a writable connection."
                )
            try:
                yield session
            finally:
                await session.rollback()
    finally:
        await engine.dispose()


async def table_counts(
    session: AsyncSession, tables: Sequence[str] = PROD_TABLES
) -> dict[str, int]:
    """Row count per table. Tables that do not exist are **omitted** from the result."""
    counts: dict[str, int] = {}
    for table in tables:
        if not _IDENTIFIER.match(table):
            raise ValueError(f"unsafe table name: {table!r}")
        if await session.scalar(text("SELECT to_regclass(:name)"), {"name": table}) is None:
            continue
        counts[table] = await session.scalar(text(f'SELECT count(*) FROM "{table}"'))
    return counts


async def alembic_revision(session: AsyncSession) -> str | None:
    """The database's current Alembic revision, or None when it has no Alembic history."""
    if await session.scalar(text("SELECT to_regclass('alembic_version')")) is None:
        return None
    return await session.scalar(text("SELECT version_num FROM alembic_version"))
