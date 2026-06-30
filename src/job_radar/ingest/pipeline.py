import asyncio
import logging
from dataclasses import asdict
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.adapters.embeddings import embed
from job_radar.adapters.sources.base import NormalizedJob, SourceAdapter
from job_radar.db.models import Job
from job_radar.ingest.dedup import content_hash
from job_radar.ingest.extract import extract_fields
from job_radar.retrieval.seniority import normalize_level

logger = logging.getLogger(__name__)

# Caps concurrent LLM + embed calls per adapter run. Matches Ollama's default
# OLLAMA_NUM_PARALLEL for GPU; on CPU Ollama still queues them so this just
# ensures we don't spin up thousands of coroutines on large batches.
_MAX_CONCURRENT_INGEST = 20


async def _prepare(
    r: NormalizedJob,
    h: str,
    ingested_via: str,
    semaphore: asyncio.Semaphore,
) -> Job | None:
    """Extract structured fields, embed, and return a Job ready for DB insert.

    All I/O is done inside the semaphore so at most _MAX_CONCURRENT_INGEST
    postings run through Ollama at once. Returns None and logs on HTTP failure.
    """
    async with semaphore:
        try:
            requirements, responsibilities = await extract_fields(r.description)
            # Embed role-specific content only; fall back to full description
            # when extraction produced nothing (LLM failure or very short posting).
            if requirements or responsibilities:
                embed_text = "\n".join(filter(None, [r.title, requirements, responsibilities]))
            else:
                embed_text = f"{r.title}\n{r.description}"
            embedding = await embed(embed_text, task="document")
            return Job(
                **asdict(r),
                ingested_via=ingested_via,
                collected_at=datetime.now(UTC),
                content_hash=h,
                embedding=embedding,
                seniority=normalize_level(r.title),
                requirements=requirements or None,
                responsibilities=responsibilities or None,
            )
        except httpx.HTTPError as exc:
            logger.warning("Skipping job %s (hash %s): %s", r.url, h, exc)
            return None


async def run_ingestion(adapter: SourceAdapter, session: AsyncSession, ingested_via: str) -> None:
    """Fetch, dedupe, embed, and persist new postings from `adapter`.

    Two-phase: extract+embed runs concurrently across all new postings
    (_MAX_CONCURRENT_INGEST at once), then DB inserts happen sequentially so
    the session is never touched from multiple coroutines. Each insert is its
    own savepoint so a constraint violation skips that posting without rolling
    back the rest.
    """
    raw_postings = await adapter.fetch()
    mapped_results = [adapter.map(raw) for raw in raw_postings]
    hashed_results = {content_hash(r): r for r in mapped_results}

    # A posting is "already seen" if either its content hash or its URL is in the
    # DB. The URL check matters when a posting's content changed (new hash, same
    # URL): without it we would re-embed and then fail the URL unique constraint
    # on insert every run. We keep the existing row rather than updating it.
    existing_hashes = set(
        (
            await session.execute(
                select(Job.content_hash).where(Job.content_hash.in_(hashed_results.keys()))
            )
        )
        .scalars()
        .all()
    )
    existing_urls = set(
        (
            await session.execute(
                select(Job.url).where(Job.url.in_(r.url for r in hashed_results.values()))
            )
        )
        .scalars()
        .all()
    )

    new_results = {
        h: r
        for h, r in hashed_results.items()
        if h not in existing_hashes and r.url not in existing_urls
    }

    if not new_results:
        return

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_INGEST)
    outcomes: list[Job | None | BaseException] = list(
        await asyncio.gather(
            *(_prepare(r, h, ingested_via, semaphore) for h, r in new_results.items()),
            return_exceptions=True,
        )
    )

    for item in outcomes:
        if isinstance(item, BaseException):
            logger.warning("Unexpected error preparing posting: %s", item)
            continue
        if item is None:
            continue
        try:
            async with session.begin_nested():
                session.add(item)
                await session.flush()
        except SQLAlchemyError as exc:
            logger.warning("Failed to insert job (hash %s): %s", item.content_hash, exc)

    await session.commit()
