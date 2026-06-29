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
) -> list[Job]:
    query_embedding = await embed(query)
    fts_results = await search_fts(session, query, pool, extra_filter)
    vector_results = await search_vector(session, query_embedding, pool, extra_filter)
    fused = reciprocal_rank_fusion([fts_results, vector_results], limit=limit)

    if not fused:
        return []

    ids = [job_id for job_id, _ in fused]
    rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars().all()
    by_id = {job.id: job for job in rows}

    return [by_id[job_id] for job_id in ids]
