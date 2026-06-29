from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job


async def search_vector(
    session: AsyncSession,
    query_embedding: list[float],
    limit: int,
    extra_filter: ColumnElement[bool] | None = None,
) -> list[tuple[UUID, float]]:
    similarity = 1 - Job.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Job.id, similarity)
        .where(Job.embedding.is_not(None))
        .order_by(similarity.desc())
        .limit(limit)
    )
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    rows = (await session.execute(stmt)).all()
    return [(job_id, float(score)) for job_id, score in rows]
