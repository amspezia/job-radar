"""Create, drop and migrate the EVALS database.

Synchronous on purpose: `CREATE DATABASE` cannot run inside a transaction (hence a raw
psycopg connection in autocommit on the `postgres` maintenance database), and
`alembic.command` is a blocking API.

Nothing here can reach production: `drop_database` refuses any name that does not contain
`_test_`, and `migrate` only ever targets the URL it is handed or `EVALS_DATABASE_URL`.
"""

import re
from pathlib import Path

import psycopg
from alembic.config import Config
from sqlalchemy.engine import make_url

from alembic import command
from eval.evals_db.settings import get_settings

# Repo root: eval/evals_db/admin.py -> eval/evals_db -> eval -> .
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic_evals.ini"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def database_name(url: str) -> str:
    name = make_url(url).database
    if not name:
        raise ValueError(f"no database name in URL: {make_url(url).render_as_string()}")
    return name


def with_database(url: str, name: str) -> str:
    """The same server and credentials, pointed at a different database."""
    return make_url(url).set(database=name).render_as_string(hide_password=False)


def _quoted(name: str) -> str:
    """`CREATE`/`DROP DATABASE` cannot be parameterized, so the name is validated then quoted."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"unsafe database name: {name!r}")
    return f'"{name}"'


def _maintenance_connection(url: str) -> psycopg.Connection:
    dsn = make_url(with_database(url, "postgres")).set(drivername="postgresql")
    return psycopg.connect(dsn.render_as_string(hide_password=False), autocommit=True)


def ensure_database(url: str) -> bool:
    """Create the database named by `url` if it does not exist. True when it was created."""
    name = database_name(url)
    statement = f"CREATE DATABASE {_quoted(name)}"
    with _maintenance_connection(url) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if exists:
            return False
        conn.execute(statement)
    return True


def drop_database(url: str) -> None:
    """Drop a throwaway database. Refuses any name without `_test_` in it."""
    name = database_name(url)
    if "_test_" not in name:
        raise ValueError(
            f"refusing to drop {name!r}: only throwaway databases (name containing '_test_') "
            "may be dropped"
        )
    statement = f"DROP DATABASE IF EXISTS {_quoted(name)} WITH (FORCE)"
    with _maintenance_connection(url) as conn:
        conn.execute(statement)


def migrate(url: str | None = None) -> None:
    """Upgrade the EVALS database to head.

    The Alembic config is located from this file, not the CWD, so `migrate()` behaves the
    same from a CLI, a test fixture or anywhere else.
    """
    target = url or get_settings().evals_database_url
    config = Config(str(ALEMBIC_INI))
    # set_main_option goes through ConfigParser interpolation, where `%` is a metacharacter.
    config.set_main_option("sqlalchemy.url", target.replace("%", "%%"))
    command.upgrade(config, "head")
