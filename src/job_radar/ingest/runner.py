import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.adapters.sources.arbeitnow import ArbeitnowAdapter
from job_radar.adapters.sources.ashby import AshbyAdapter
from job_radar.adapters.sources.getonboard import GetOnBoardAdapter
from job_radar.adapters.sources.greenhouse import GreenHouseAdapter
from job_radar.adapters.sources.himalayas import HimalayasAdapter
from job_radar.adapters.sources.lever import LeverAdapter
from job_radar.adapters.sources.remotive import RemotiveAdapter
from job_radar.adapters.sources.smartrecruiters import SmartRecruitersAdapter
from job_radar.adapters.sources.workable import WorkableAdapter
from job_radar.ingest.pipeline import run_ingestion

logger = logging.getLogger(__name__)

ENABLED_ADAPTERS = [
    RemotiveAdapter(),
    ArbeitnowAdapter(),
    HimalayasAdapter(),
    GreenHouseAdapter(),
    GetOnBoardAdapter(),
    LeverAdapter(),
    AshbyAdapter(),
    WorkableAdapter(),
    SmartRecruitersAdapter(),
]


async def run_all_ingestion(session: AsyncSession, ingested_via: str) -> None:
    """Run ingestion for every enabled adapter.

    Each adapter's failure is isolated and logged so one source being down
    doesn't prevent the others from running.

    fetch() for the *next* adapter is kicked off before this adapter's
    extract+embed phase runs, so its network I/O overlaps with the current
    adapter's LLM-bound work instead of waiting behind it — measured on this
    project's adapters, fetch is pure network I/O with no DB session access,
    while extract+embed is pure LLM I/O with no session access either, so the
    two overlap safely. Only the insert loop inside run_ingestion touches the
    session, and that still runs strictly sequentially per adapter.
    """
    fetch_task = asyncio.create_task(ENABLED_ADAPTERS[0].fetch())
    for i, adapter in enumerate(ENABLED_ADAPTERS):
        logger.info("Starting ingestion for source=%s", adapter.source)
        try:
            raw_postings = await fetch_task
        except Exception:
            logger.exception("Fetch failed for source=%s", adapter.source)
            raw_postings = []
        if i + 1 < len(ENABLED_ADAPTERS):
            fetch_task = asyncio.create_task(ENABLED_ADAPTERS[i + 1].fetch())
        try:
            await run_ingestion(
                adapter, session, ingested_via=ingested_via, raw_postings=raw_postings
            )
        except Exception:
            logger.exception("Ingestion failed for source=%s", adapter.source)
            continue
        logger.info("Finished ingestion for source=%s", adapter.source)
