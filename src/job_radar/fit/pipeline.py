import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job, Profile
from job_radar.fit.analyze import analyze_fit
from job_radar.fit.schema import FitAssessment
from job_radar.retrieval.search import search

logger = logging.getLogger(__name__)


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
    logger.info("Searching with query=%r", query)
    jobs = await search(session, query, limit=limit)
    logger.info("Retrieved %d candidate jobs", len(jobs))

    results = [(job, await analyze_fit(profile, job)) for job in jobs]
    results.sort(key=lambda pair: pair[1].score if pair[1].score is not None else -1, reverse=True)
    return results
