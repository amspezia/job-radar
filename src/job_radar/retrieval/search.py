from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job
from job_radar.retrieval.bm25 import search_bm25
from job_radar.retrieval.fusion import reciprocal_rank_fusion
from job_radar.retrieval.vector import search_vector


async def search(
    session: AsyncSession,
    query: str,
    *,
    hyde_embedding: list[float] | None = None,
    limit: int = 20,
    pool: int = 50,
    extra_filter: ColumnElement[bool] | None = None,
    profile_embedding: list[float] | None = None,
    weights: list[float] | None = None,
) -> list[Job]:
    """Hybrid search fusing up to three pre-computed rankers via RRF.

    Arms, each contributing only when it has signal:
    - lexical (BM25) over `query` — keyword bag, title-boosted,
    - HyDE — cosine similarity against a synthetic job posting embedded as a
      document (same space as indexed descriptions; caller pre-computes),
    - CV — cosine similarity against the candidate's CV embedding.

    BM25 + HyDE are skipped when `query` is blank and `hyde_embedding` is None.
    With no arms at all the result is empty rather than an unfiltered corpus dump.

    `weights` must have the same length as the number of active arms when provided;
    defaults to equal weights (standard RRF). Values are tuned by the eval phase.
    """
    arms: list[list[tuple[UUID, float]]] = []

    if query and query.strip():
        arms.append(await search_bm25(session, query, pool, extra_filter))

    if hyde_embedding is not None:
        arms.append(await search_vector(session, hyde_embedding, pool, extra_filter))

    if profile_embedding is not None:
        arms.append(await search_vector(session, profile_embedding, pool, extra_filter))

    fused = reciprocal_rank_fusion(arms, limit=limit, weights=weights)
    if not fused:
        return []

    ids = [job_id for job_id, _ in fused]
    rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars().all()
    by_id = {job.id: job for job in rows}

    return [by_id[job_id] for job_id in ids]
