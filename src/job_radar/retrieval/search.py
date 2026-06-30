from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.adapters.embeddings import embed
from job_radar.db.models import Job
from job_radar.retrieval.fts import search_fts
from job_radar.retrieval.fusion import reciprocal_rank_fusion
from job_radar.retrieval.vector import search_vector


async def search(
    session: AsyncSession,
    query: str,
    *,
    limit: int = 20,
    pool: int = 50,
    extra_filter: ColumnElement[bool] | None = None,
    profile_embedding: list[float] | None = None,
) -> list[Job]:
    """Hybrid search fusing up to three rankers via RRF.

    Arms, each contributing only when it has signal:
    - lexical (FTS) over the query text,
    - semantic over the embedded query text,
    - semantic over a precomputed profile/CV embedding (candidate-anchored).

    The first two are skipped when the query is blank; the CV arm is skipped when
    no embedding is supplied. With no arms (blank query and no embedding) the
    result is empty rather than an unfiltered dump.
    """
    arms: list[list[tuple[UUID, float]]] = []

    if query and query.strip():
        arms.append(await search_fts(session, query, pool, extra_filter))
        arms.append(await search_vector(session, await embed(query), pool, extra_filter))

    if profile_embedding is not None:
        arms.append(await search_vector(session, profile_embedding, pool, extra_filter))

    fused = reciprocal_rank_fusion(arms, limit=limit)
    if not fused:
        return []

    ids = [job_id for job_id, _ in fused]
    rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars().all()
    by_id = {job.id: job for job in rows}

    return [by_id[job_id] for job_id in ids]
