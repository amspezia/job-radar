"""What a ranking method ranks, and what every ranking method must be."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.models import EmbeddingJob, EmbeddingTopic, EmbeddingTopicJob


@dataclass(frozen=True)
class CorpusDoc:
    """One frozen document. `embed_text` is production's representation, rebuilt at import."""

    id: UUID
    embed_text: str
    title: str
    description: str
    requirements: str | None
    responsibilities: str | None
    source: str
    seniority: str | None


@dataclass
class TopicView:
    """A topic as a ranking method sees it: the query inputs and the closed corpus."""

    topic_id: UUID
    name: str
    query_inputs: dict
    profile_snapshot: dict
    docs: list[CorpusDoc]


async def load_topic_view(session: AsyncSession, name: str) -> TopicView:
    """Load a topic and its corpus, ordered by job id so every run sees the same order."""
    topic = (
        await session.execute(select(EmbeddingTopic).where(EmbeddingTopic.name == name))
    ).scalar_one_or_none()
    if topic is None:
        raise LookupError(f"no topic named {name!r}")

    rows = await session.execute(
        select(
            EmbeddingJob.id,
            EmbeddingJob.embed_text,
            EmbeddingJob.title,
            EmbeddingJob.description,
            EmbeddingJob.requirements,
            EmbeddingJob.responsibilities,
            EmbeddingJob.source,
            EmbeddingJob.seniority,
        )
        .join(EmbeddingTopicJob, EmbeddingTopicJob.job_id == EmbeddingJob.id)
        .where(EmbeddingTopicJob.topic_id == topic.id)
        .order_by(EmbeddingJob.id)
    )
    return TopicView(
        topic_id=topic.id,
        name=topic.name,
        query_inputs=topic.query_inputs,
        profile_snapshot=topic.profile_snapshot,
        docs=[CorpusDoc(*row) for row in rows],
    )


class RankingMethod(Protocol):
    """Anything the runner can run: dense today, hybrid or a reranker later."""

    name: str

    def fingerprint(self) -> str: ...
    def describe(self) -> dict: ...
    async def prepare(self, topic: TopicView) -> None: ...
    def rank(self, topic: TopicView) -> list[tuple[UUID, float]]: ...
    def stats(self) -> dict: ...
