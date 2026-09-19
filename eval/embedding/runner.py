"""Run a ranking method over a topic and store its full ranking."""

import time
from datetime import UTC, datetime
from itertools import batched
from uuid import UUID, uuid4

from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.methods.base import RankingMethod, load_topic_view
from eval.embedding.models import EmbeddingRun, EmbeddingRunRanking

_RANKING_CHUNK = 1000


async def run_method(
    session: AsyncSession,
    topic_name: str,
    method: RankingMethod,
    git_sha: str | None = None,
) -> UUID:
    """Rank `topic_name` with `method`, storing the run and every ranked document.

    Knows nothing about embedders: a method's `stats()` contributes whatever it has. The run
    row is committed before the work starts, so a run that dies is visible as `failed` rather
    than absent.
    """
    view = await load_topic_view(session, topic_name)
    run_id = uuid4()
    session.add(
        EmbeddingRun(
            id=run_id,
            topic_id=view.topic_id,
            name=method.name,
            method_fingerprint=method.fingerprint(),
            method_config=method.describe(),
            git_sha=git_sha,
            status="running",
        )
    )
    await session.commit()

    started = time.perf_counter()
    try:
        await method.prepare(view)
        ranking = method.rank(view)
        for chunk in batched(enumerate(ranking, start=1), _RANKING_CHUNK):
            await session.execute(
                insert(EmbeddingRunRanking),
                [
                    {"run_id": run_id, "job_id": job_id, "rank": rank, "score": score}
                    for rank, (job_id, score) in chunk
                ],
            )
        stats = method.stats()
        await session.execute(
            update(EmbeddingRun)
            .where(EmbeddingRun.id == run_id)
            .values(
                status="completed",
                # Only known after prepare(): the embedder row it references is written there.
                embedder_fingerprint=stats.get("embedder_fingerprint"),
                docs_per_s=stats.get("docs_per_s"),
                n_truncated=stats.get("n_truncated"),
                seconds=time.perf_counter() - started,
                finished_at=datetime.now(UTC),
            )
        )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        await session.execute(
            update(EmbeddingRun)
            .where(EmbeddingRun.id == run_id)
            .values(
                status="failed",
                error=str(exc)[:2000],
                seconds=time.perf_counter() - started,
                finished_at=datetime.now(UTC),
            )
        )
        await session.commit()
        raise
    return run_id
