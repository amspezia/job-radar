"""Alembic environment for the EVALS database — see `alembic_evals.ini`.

Mirrors production's async `alembic/env.py` (NullPool + `run_sync`) but targets
`EvalsBase.metadata` and must never import `job_radar`: the harness is isolated from
production code (implementation plan §3, enforced by tests/test_evals_isolation.py).

Offline mode is not implemented — nothing in this project generates SQL scripts.
"""

import asyncio

from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from eval.embedding import models  # noqa: F401  (registers the embedding_* tables)
from eval.evals_db.base import EvalsBase
from eval.evals_db.settings import get_settings

config = context.config
# Set by admin.migrate() and by the test fixtures; a bare `alembic -c alembic_evals.ini`
# falls back to the configured EVALS database. `%` is a ConfigParser metacharacter.
url = config.get_main_option("sqlalchemy.url") or get_settings().evals_database_url
config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))

target_metadata = EvalsBase.metadata


def run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(run_migrations)
    await connectable.dispose()


asyncio.run(run_async_migrations())
