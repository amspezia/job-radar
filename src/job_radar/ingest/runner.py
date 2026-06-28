import logging

from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.ingest.adapters.arbeitnow import ArbeitnowAdapter
from job_radar.ingest.adapters.getonboard import GetOnBoardAdapter
from job_radar.ingest.adapters.greenhouse import GreenHouseAdapter
from job_radar.ingest.adapters.himalayas import HimalayasAdapter
from job_radar.ingest.adapters.remotive import RemotiveAdapter
from job_radar.ingest.pipeline import run_ingestion

logger = logging.getLogger(__name__)

ENABLED_ADAPTERS = [
    RemotiveAdapter(),
    ArbeitnowAdapter(),
    HimalayasAdapter(),
    GreenHouseAdapter(),
    GetOnBoardAdapter(),
]


async def run_all_ingestion(session: AsyncSession, ingested_via: str) -> None:
    """Run ingestion for every enabled adapter.

    Each adapter's failure is isolated and logged so one source being down
    doesn't prevent the others from running.
    """
    for adapter in ENABLED_ADAPTERS:
        logger.info("Starting ingestion for source=%s", adapter.source)
        try:
            await run_ingestion(adapter, session, ingested_via=ingested_via)
        except Exception:
            logger.exception("Ingestion failed for source=%s", adapter.source)
            continue
        logger.info("Finished ingestion for source=%s", adapter.source)
