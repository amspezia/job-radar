from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job

_FTS_CONFIG = "simple"


async def search_fts(
    session: AsyncSession,
    query_text: str,
    limit: int,
    extra_filter: ColumnElement[bool] | None = None,
) -> list[tuple[UUID, float]]:
    """Full-text search over jobs, ranked by ts_rank descending.

    Returns (job_id, score) pairs.

    websearch_to_tsquery parses Google-style syntax (bare words -> implicit
    AND, "phrase" -> proximity match, -word -> negation)
    """
    tsquery = func.websearch_to_tsquery(_FTS_CONFIG, query_text)
    rank = func.ts_rank(Job.search_vector, tsquery)
    stmt = (
        select(Job.id, rank)
        .where(Job.search_vector.bool_op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    rows = (await session.execute(stmt)).all()
    return [(job_id, float(score)) for job_id, score in rows]
