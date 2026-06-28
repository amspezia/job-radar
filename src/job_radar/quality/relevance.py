from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job

# Canonical developer/engineering role phrases. Their averaged embedding defines
# the "engineering" direction every stored posting vector is scored against.
ANCHORS = [
    "software engineer",
    "backend developer",
    "frontend developer",
    "full stack developer",
    "data engineer",
    "data scientist",
    "machine learning engineer",
    "devops engineer",
    "platform engineer",
    "mobile developer",
]

# Cosine threshold above which a posting counts as "on scope". A starting point —
# the per-source relevance_mean the report prints is the calibration aid.
DEFAULT_THRESHOLD = 0.5

EmbedFn = Callable[[str], Awaitable[list[float]]]


@dataclass
class SourceRelevance:
    relevance_mean: float
    dev_embed_pct: float


async def build_centroid(embed: EmbedFn, anchors: list[str] = ANCHORS) -> list[float]:
    """Embed each anchor with the project's own embed() and average the vectors.

    Using the same embed() as ingestion guarantees the centroid lives in the
    same space as the stored job vectors, regardless of model/prefix specifics.
    """
    vectors = [await embed(anchor) for anchor in anchors]
    count = len(vectors)
    return [sum(v[i] for v in vectors) / count for i in range(len(vectors[0]))]


async def relevance_by_source(
    session: AsyncSession, centroid: list[float], threshold: float
) -> dict[str, SourceRelevance]:
    """Per-source mean cosine-similarity to the centroid and on-scope rate.

    The cosine is pushed into Postgres via pgvector's distance operator, so no
    embeddings are pulled into Python.
    """
    similarity = 1 - Job.embedding.cosine_distance(centroid)
    stmt = (
        select(
            Job.source,
            func.avg(similarity),
            func.avg(case((similarity >= threshold, 1.0), else_=0.0)),
        )
        .where(Job.embedding.is_not(None))
        .group_by(Job.source)
    )
    rows = (await session.execute(stmt)).all()
    return {
        source: SourceRelevance(round(float(mean), 4), round(100.0 * float(pct), 1))
        for source, mean, pct in rows
    }
