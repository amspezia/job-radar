import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from job_radar.db.base import async_session_factory
from job_radar.ingest.runner import run_all_ingestion

logger = logging.getLogger(__name__)


async def _scheduled_ingestion() -> None:
    async with async_session_factory() as session:
        await run_all_ingestion(session, ingested_via="scheduler")


def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _scheduled_ingestion,
        trigger=CronTrigger(hour=3, minute=0),
        id="daily_ingestion",
        misfire_grace_time=10800,  # 3h: still run after a restart/outage near 03:00.
        coalesce=True,  # if multiple fires were missed, run once to catch up, not N times.
        max_instances=1,  # never let two ingestion runs overlap.
    )
    return scheduler


async def _main() -> None:
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("Scheduler started; daily ingestion scheduled for 03:00.")
    try:
        await asyncio.Event().wait()
    finally:
        scheduler.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())


if __name__ == "__main__":
    main()
