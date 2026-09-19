"""What the read-only guard does, and — just as important — what it does not do.

The "production" database in every test here is the *throwaway EVALS database*: the guard
mechanism is identical, and a test that probes writes must never be pointed at the real
thing. Nothing in this file opens the `DATABASE_URL` database.
"""

import os
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from eval.evals_db import admin, prod_reader
from eval.evals_db.prod_reader import ProdNotReadOnly, alembic_revision, prod_session, table_counts

# Each of these aborts its transaction, so each runs in a session of its own.
WRITES = (
    "CREATE TABLE _ro_probe_x (id integer)",
    "INSERT INTO embedding_topic (id, tier, name, status, profile_snapshot, query_inputs, builder)"
    " VALUES (gen_random_uuid(), 'A', 'probe', 'draft', '{}', '{}', 'probe')",
    "UPDATE embedding_topic SET name = name WHERE false",
    "DELETE FROM embedding_topic WHERE false",
    # Not even against a table that does not exist: the read-only check fires first.
    "TRUNCATE _ro_probe_missing",
)


class _NoProdSettings:
    database_url = None


async def test_the_session_reports_itself_read_only(evals_db_url: str) -> None:
    async with prod_session(evals_db_url) as session:
        assert await session.scalar(text("SHOW transaction_read_only")) == "on"
        assert await session.scalar(text("SELECT count(*) FROM embedding_topic")) == 0


@pytest.mark.parametrize("statement", WRITES)
async def test_writes_are_rejected(evals_db_url: str, statement: str) -> None:
    with pytest.raises(DBAPIError, match="read-only transaction"):
        async with prod_session(evals_db_url) as session:
            await session.execute(text(statement))


async def test_a_rejected_write_leaves_nothing_behind(evals_db_url: str) -> None:
    async with prod_session(evals_db_url) as session:
        assert await session.scalar(text("SELECT to_regclass('_ro_probe_x')")) is None


async def test_an_engine_without_the_option_is_refused(
    evals_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def writable_engine(url: str, **kwargs: object):
        kwargs.pop("connect_args", None)
        return create_async_engine(url, poolclass=NullPool)

    monkeypatch.setattr(prod_reader, "create_async_engine", writable_engine)

    with pytest.raises(ProdNotReadOnly, match="'off'"):
        async with prod_session(evals_db_url):
            pass


async def test_a_connection_whose_default_was_turned_off_is_refused(
    evals_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `SET default_transaction_read_only = off` (or `SET TRANSACTION READ WRITE`) inside a
    # session defeats the connect option — the guard stops accidents, not a deliberate
    # override (review R-m4). What it does catch is a connection that arrives in that
    # state, and this is the check that catches it.
    def overridden_engine(url: str, **kwargs: object):
        return create_async_engine(
            url,
            poolclass=NullPool,
            connect_args={"options": "-c default_transaction_read_only=off"},
        )

    monkeypatch.setattr(prod_reader, "create_async_engine", overridden_engine)

    with pytest.raises(ProdNotReadOnly):
        async with prod_session(evals_db_url):
            pass


async def test_table_counts_skips_tables_that_do_not_exist(evals_db_url: str) -> None:
    async with prod_session(evals_db_url) as session:
        counts = await table_counts(session, ("embedding_topic", "embedding_label", "jobs"))

    assert counts == {"embedding_topic": 0, "embedding_label": 0}


async def test_table_counts_refuses_a_name_it_cannot_quote(evals_db_url: str) -> None:
    async with prod_session(evals_db_url) as session:
        with pytest.raises(ValueError, match="unsafe table name"):
            await table_counts(session, ('jobs"; DROP TABLE jobs; --',))


async def test_alembic_revision_reads_the_history(evals_db_url: str) -> None:
    config = Config(str(admin.ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", evals_db_url.replace("%", "%%"))
    head = ScriptDirectory.from_config(config).get_current_head()

    async with prod_session(evals_db_url) as session:
        assert await alembic_revision(session) == head


async def test_alembic_revision_is_none_without_a_history() -> None:
    url = admin.with_database(os.environ["DATABASE_URL"], f"job_radar_evals_test_{uuid4().hex[:8]}")
    admin.ensure_database(url)  # created, deliberately never migrated
    try:
        async with prod_session(url) as session:
            assert await alembic_revision(session) is None
    finally:
        admin.drop_database(url)


async def test_an_unset_production_url_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prod_reader, "get_settings", _NoProdSettings)

    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        async with prod_session():
            pass
