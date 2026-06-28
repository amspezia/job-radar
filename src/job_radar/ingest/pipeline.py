import logging
from dataclasses import asdict
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job
from job_radar.ingest.base import SourceAdapter
from job_radar.ingest.embed import embed
from job_radar.ingest.normalize import content_hash

logger = logging.getLogger(__name__)


async def run_ingestion(adapter: SourceAdapter, session: AsyncSession, ingested_via: str) -> None:
    """Fetch, dedupe, embed, and persist new postings from `adapter`.

    Commits internally. Each posting is embedded and inserted in its own
    savepoint, so an embedding-call failure or a constraint violation skips
    that one posting (logged) rather than losing the rest of the batch.
    """
    raw_postings = await adapter.fetch()
    mapped_results = [adapter.map(raw) for raw in raw_postings]
    hashed_results = {content_hash(r): r for r in mapped_results}

    existing_hashes = set(
        (
            await session.execute(
                select(Job.content_hash).where(Job.content_hash.in_(hashed_results.keys()))
            )
        )
        .scalars()
        .all()
    )

    new_results = {h: r for h, r in hashed_results.items() if h not in existing_hashes}

    for h, r in new_results.items():
        try:
            embedding = await embed(f"{r.title}\n{r.description}")
            async with session.begin_nested():
                session.add(
                    Job(
                        **asdict(r),
                        ingested_via=ingested_via,
                        collected_at=datetime.now(UTC),
                        content_hash=h,
                        embedding=embedding,
                    )
                )
                await session.flush()
        except (httpx.HTTPError, IntegrityError) as exc:
            logger.warning("Skipping job %s (hash %s): %s", r.url, h, exc)
            continue

    await session.commit()
