import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job, Profile
from job_radar.fit.analyze import analyze_fit
from job_radar.fit.schema import FitAssessment
from job_radar.retrieval.geo import build_geo_filter
from job_radar.retrieval.search import search

logger = logging.getLogger(__name__)

# Caps how many analyze_fit calls run at once. Matches Ollama's typical default
# OLLAMA_NUM_PARALLEL; unbounded concurrency would just queue identically to
# sequential (or exhaust GPU VRAM) instead of actually overlapping.
_MAX_CONCURRENT_ANALYSES = 4


async def _load_profile(session: AsyncSession) -> Profile | None:
    return (await session.execute(select(Profile))).scalars().first()


def default_query(profile: Profile) -> str:
    """Build a retrieval query from the profile when the caller has none."""
    return " ".join(profile.target_titles or [])


async def run_fit_pipeline(
    session: AsyncSession, query: str | None = None, *, limit: int = 20
) -> list[tuple[Job, FitAssessment]]:
    """Retrieve candidate jobs for the stored profile and score each one's fit.

    Results are sorted best-first; jobs the pre-flight guard skipped (score is
    None) sort last.
    """
    profile = await _load_profile(session)
    if profile is None:
        raise ValueError("no profile loaded — run job-radar-profile first")

    query = query or default_query(profile)
    keywords = (profile.location_rules or {}).get("allowed_keywords")
    geo_filter = build_geo_filter(keywords) if keywords else None
    logger.info("Searching with query=%r geo_filtered=%s", query, geo_filter is not None)
    jobs = await search(session, query, limit=limit, extra_filter=geo_filter)
    logger.info("Retrieved %d candidate jobs", len(jobs))

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)

    async def _bounded_analyze(job: Job) -> tuple[Job, FitAssessment]:
        async with semaphore:
            return job, await analyze_fit(profile, job)

    results = await asyncio.gather(*(_bounded_analyze(job) for job in jobs))
    return sorted(
        results, key=lambda pair: pair[1].score if pair[1].score is not None else -1, reverse=True
    )
