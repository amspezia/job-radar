"""`job-radar-evals-db` — create, migrate and inspect the EVALS database."""

import argparse
import asyncio

from sqlalchemy import text

from eval.evals_db import admin
from eval.evals_db.base import EvalsBase, evals_session
from eval.evals_db.settings import get_settings


async def _status(url: str) -> None:
    # Imported for its side effect: it puts the embedding_* tables on EvalsBase.metadata.
    import eval.embedding.models  # noqa: F401

    async with evals_session(url) as session:
        revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
        print(f"{admin.database_name(url)} @ {revision or 'no revision'}")
        for table in EvalsBase.metadata.sorted_tables:
            count = await session.scalar(text(f'SELECT count(*) FROM "{table.name}"'))
            print(f"  {table.name:<24} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="job-radar-evals-db", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the EVALS database if absent, then migrate it")
    commands.add_parser("migrate", help="upgrade the EVALS database to head")
    commands.add_parser("status", help="print the current revision and the embedding_* row counts")
    args = parser.parse_args()

    url = get_settings().evals_database_url
    if args.command == "init":
        created = admin.ensure_database(url)
        print(f"{'created' if created else 'already present'}: {admin.database_name(url)}")
        admin.migrate(url)
        asyncio.run(_status(url))
    elif args.command == "migrate":
        admin.migrate(url)
        asyncio.run(_status(url))
    else:
        asyncio.run(_status(url))
