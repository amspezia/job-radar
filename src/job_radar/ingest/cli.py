import asyncio
import logging

from job_radar.db.base import async_session_factory
from job_radar.ingest.runner import run_all_ingestion


async def _main() -> None:
    async with async_session_factory() as session:
        await run_all_ingestion(session, ingested_via="scheduler")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())


if __name__ == "__main__":
    main()
