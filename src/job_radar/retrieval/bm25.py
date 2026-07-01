from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job


def _boosted_query(query_text: str, field_boosts: dict[str, int] | None = None) -> str:
    # Three indexed fields, each with its own BM25 length normalization:
    #   title       5x — highest discriminative signal; short, so it never
    #               gets length-penalised relative to the description blob.
    #   requirements 2x — tech stack terms land here; higher weight than
    #               responsibilities because stack keywords are more specific.
    #   responsibilities 1x — baseline; provides recall for role verbs.
    # lenient=true silences parse errors from special chars (C++, .NET, etc.)
    boosts = field_boosts or {"title": 5, "requirements": 3, "responsibilities": 1}
    return " ".join(f"{field}:({query_text})^{weight}" for field, weight in boosts.items())


async def search_bm25(
    session: AsyncSession,
    query_text: str,
    limit: int,
    extra_filter: ColumnElement[bool] | None = None,
    *,
    field_boosts: dict[str, int] | None = None,
) -> list[tuple[UUID, float]]:
    """BM25 search over jobs via ParadeDB pg_search.

    Title matches are boosted 5x over description at query time. BM25 scoring
    provides IDF + length normalisation that ts_rank lacks: rare discriminative
    terms (e.g. "Kafka") outweigh ubiquitous ones ("engineer"), and partial
    keyword matches score without requiring every query token to appear.

    Returns (job_id, score) pairs ordered best-first.
    """
    score = func.paradedb.score(Job.id)
    stmt = (
        select(Job.id, score)
        .where(
            text("jobs @@@ paradedb.parse(:q, lenient => true)").bindparams(
                q=_boosted_query(query_text, field_boosts)
            )
        )
        .order_by(score.desc())
        .limit(limit)
    )
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    rows = (await session.execute(stmt)).all()
    return [(job_id, float(s)) for job_id, s in rows]
