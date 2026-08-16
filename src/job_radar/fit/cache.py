import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import FitJudgmentCache, Job
from job_radar.fit.analyze import PROMPT_VERSION
from job_radar.fit.schema import FitJudgment

logger = logging.getLogger(__name__)


async def load(
    session: AsyncSession, profile_id: UUID, jobs: list[Job], *, model: str
) -> dict[UUID, FitJudgment]:
    """Fetch judgments already generated for these jobs under the current cache key.

    Keyed on the job's *current* content_hash, so an edited posting misses rather
    than returning a judgment of text that no longer exists. A row whose stored
    judgment no longer validates is treated as a miss — a schema change that
    slipped past PROMPT_VERSION must cost a re-analysis, never a crash.
    """
    if not jobs:
        return {}

    hash_by_id = {job.id: job.content_hash for job in jobs}
    rows = (
        (
            await session.execute(
                select(FitJudgmentCache).where(
                    FitJudgmentCache.profile_id == profile_id,
                    FitJudgmentCache.job_id.in_(hash_by_id),
                    FitJudgmentCache.model == model,
                    FitJudgmentCache.prompt_version == PROMPT_VERSION,
                )
            )
        )
        .scalars()
        .all()
    )

    cached: dict[UUID, FitJudgment] = {}
    for row in rows:
        if row.content_hash != hash_by_id[row.job_id]:
            continue  # posting changed since it was judged
        try:
            cached[row.job_id] = FitJudgment.model_validate(row.judgment)
        except Exception:
            logger.warning("Discarding unreadable cached judgment for job %s", row.job_id)
    return cached


async def store(
    session: AsyncSession,
    profile_id: UUID,
    judgments: list[tuple[Job, FitJudgment]],
    *,
    model: str,
) -> None:
    """Persist freshly generated judgments, overwriting any row on the same key.

    ON CONFLICT DO UPDATE rather than DO NOTHING so `--refresh` genuinely replaces
    what it recomputed.
    """
    if not judgments:
        return

    now = datetime.now(UTC)
    stmt = insert(FitJudgmentCache).values(
        [
            {
                "profile_id": profile_id,
                "job_id": job.id,
                "content_hash": job.content_hash,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "judgment": judgment.model_dump(),
                "created_at": now,
            }
            for job, judgment in judgments
        ]
    )
    await session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_fit_judgments_cache_key",
            set_={"judgment": stmt.excluded.judgment, "created_at": stmt.excluded.created_at},
        )
    )
    await session.commit()
