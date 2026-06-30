import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job, Profile
from job_radar.fit.analyze import analyze_fit
from job_radar.fit.schema import FitAssessment
from job_radar.retrieval.filters import build_profile_filter
from job_radar.retrieval.search import search

logger = logging.getLogger(__name__)

# Caps how many analyze_fit calls run at once. Matches Ollama's typical default
# OLLAMA_NUM_PARALLEL; unbounded concurrency would just queue identically to
# sequential (or exhaust GPU VRAM) instead of actually overlapping.
_MAX_CONCURRENT_ANALYSES = 4


async def _load_profile(session: AsyncSession) -> Profile | None:
    return (await session.execute(select(Profile))).scalars().first()


def build_query(profile: Profile) -> str:
    """Build a retrieval query from the profile when the caller has none.

    Combines target titles with the candidate's stack and domains so the lexical
    and query-vector arms reflect the whole profile, not just the job title.
    """
    keywords = profile.domains_keywords or {}
    parts = [
        *(profile.target_titles or []),
        *keywords.get("tech_stack", []),
        *keywords.get("domains", []),
    ]
    return " ".join(parts)


async def run_fit_pipeline(
    session: AsyncSession,
    query: str | None = None,
    *,
    limit: int = 20,
    levels: list[str] | None = None,
) -> list[tuple[Job, FitAssessment]]:
    """Retrieve candidate jobs for the stored profile and score each one's fit.

    Results are sorted best-first; jobs the pre-flight guard skipped (score is
    None) sort last. `levels` overrides the profile's accepted seniority levels
    for this run, filtering retrieval and gating scoring alike.
    """
    profile = await _load_profile(session)
    if profile is None:
        raise ValueError("no profile loaded — run job-radar-profile first")

    query = query or build_query(profile)
    profile_filter = build_profile_filter(profile, levels=levels)
    logger.info("Searching with query=%r filtered=%s", query, profile_filter is not None)
    jobs = await search(
        session,
        query,
        limit=limit,
        extra_filter=profile_filter,
        profile_embedding=profile.cv_embedding,
    )
    logger.info("Retrieved %d candidate jobs", len(jobs))

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)

    async def _bounded_analyze(job: Job) -> tuple[Job, FitAssessment]:
        async with semaphore:
            return job, await analyze_fit(profile, job, levels=levels)

    results = await asyncio.gather(*(_bounded_analyze(job) for job in jobs))
    return sorted(
        results, key=lambda pair: pair[1].score if pair[1].score is not None else -1, reverse=True
    )
