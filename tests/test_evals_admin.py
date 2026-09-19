"""Creating, dropping and migrating throwaway EVALS databases.

Every database named here contains `_test_`, which is also the only kind `drop_database`
will remove. The production database is never opened.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from alembic import command
from eval.evals_db import admin, base


def _throwaway_url() -> str:
    return admin.with_database(
        os.environ["DATABASE_URL"], f"job_radar_evals_test_{uuid4().hex[:8]}"
    )


def _config(url: str) -> Config:
    config = Config(str(admin.ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def test_with_database_keeps_the_server_and_swaps_the_name() -> None:
    url = admin.with_database("postgresql+psycopg://u:p@host:5432/job_radar", "job_radar_evals")

    assert url == "postgresql+psycopg://u:p@host:5432/job_radar_evals"
    assert admin.database_name(url) == "job_radar_evals"


def test_ensure_database_is_idempotent() -> None:
    url = _throwaway_url()

    assert admin.ensure_database(url) is True
    try:
        assert admin.ensure_database(url) is False
    finally:
        admin.drop_database(url)
    # It really went away: a fresh create reports the database as new again.
    assert admin.ensure_database(url) is True
    admin.drop_database(url)


def test_drop_database_refuses_anything_but_a_throwaway() -> None:
    for name in ("job_radar", "job_radar_evals"):
        with pytest.raises(ValueError, match="_test_"):
            admin.drop_database(admin.with_database(os.environ["DATABASE_URL"], name))


def test_a_name_that_is_not_a_bare_identifier_is_refused() -> None:
    # `CREATE DATABASE` cannot be parameterized, so the name is the injection surface.
    url = admin.with_database(os.environ["DATABASE_URL"], 'evals_test_"; DROP DATABASE job_radar')

    with pytest.raises(ValueError, match="unsafe database name"):
        admin.ensure_database(url)


async def test_migrate_reaches_head(evals_db_url: str) -> None:
    head = ScriptDirectory.from_config(_config(evals_db_url)).get_current_head()

    async with base.evals_session(evals_db_url) as session:
        assert await session.scalar(text("SELECT version_num FROM alembic_version")) == head
        assert await session.scalar(text("SELECT to_regclass('embedding_job') IS NOT NULL"))


def test_alembic_check_reports_no_drift(evals_db_url: str) -> None:
    # Guards the pair `eval/embedding/models.py` <-> `alembic_evals/versions/`: a model
    # change without a migration fails here. The worker thread is what Alembic's async
    # env.py needs, since it calls asyncio.run().
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(command.check, _config(evals_db_url)).result()
