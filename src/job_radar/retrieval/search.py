import asyncio
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
    limit: int = 50,
    pool: int = 100,
    bm25_pool: int | None = None,
    extra_filter: ColumnElement[bool] | None = None,
    profile_embedding: list[float] | None = None,
    weights: list[float] | None = None,
    field_boosts: dict[str, int] | None = None,
) -> list[Job]:
    """Hybrid search fusing up to three pre-computed rankers via RRF.

    Arms, each contributing only when it has signal:
    - lexical (BM25) over `query` — keyword bag, title-boosted,
    - HyDE — cosine similarity against a synthetic job posting embedded as a
      document (same space as indexed descriptions; caller pre-computes),
    - CV — cosine similarity against the candidate's CV embedding.

    BM25 + HyDE are skipped when `query` is blank and `hyde_embedding` is None.
    With no arms at all the result is empty rather than an unfiltered corpus dump.

    `bm25_pool` independently sets the BM25 candidate pool; defaults to `pool`
    when None. Use a higher value to give BM25 more recall headroom without
    expanding the (GPU-bound) vector arm pools.

    `weights` must have the same length as the number of active arms when provided;
    defaults to equal weights (standard RRF). Values are tuned by the eval phase.
    """
    # Build arm coroutines in a fixed order (lexical=0, HyDE=1, CV=2) and track
    # which indices are active so the caller's weight vector can be sliced to
    # match exactly the arms that run — avoids a length mismatch in RRF.
    _bm25_pool = bm25_pool if bm25_pool is not None else pool
    active: list[int] = []
    coros = []
    if query and query.strip():
        active.append(0)
        coros.append(search_bm25(session, query, _bm25_pool, extra_filter, field_boosts=field_boosts))
    if hyde_embedding is not None:
        active.append(1)
        coros.append(search_vector(session, hyde_embedding, pool, extra_filter))
    if profile_embedding is not None:
        active.append(2)
        coros.append(search_vector(session, profile_embedding, pool, extra_filter))

    if not coros:
        return []

    arms: list[list[tuple[UUID, float]]] = list(await asyncio.gather(*coros))
    effective_weights = [weights[i] for i in active] if weights is not None else None
    fused = reciprocal_rank_fusion(arms, limit=limit, weights=effective_weights)
    if not fused:
        return []

    ids = [job_id for job_id, _ in fused]
    rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars().all()
    by_id = {job.id: job for job in rows}

    return [by_id[job_id] for job_id in ids]
