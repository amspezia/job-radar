"""Tier-agnostic topic framework: the payload a builder produces and how it is persisted.

The contract is docs/plans/EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md §7.2. Nothing here knows
where a payload came from, so a new tier is a new builder and no change to this module.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid4, uuid5

from sqlalchemy import insert as core_insert
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.labels import SOURCES
from eval.embedding.models import EmbeddingJob, EmbeddingLabel, EmbeddingTopic, EmbeddingTopicJob

# Fixed forever: changing it re-identifies every stored document.
_NAMESPACE = UUID("6f1d3a52-8c4e-4b7a-9d21-5e0b7c3a94f8")

# 768-d vectors are ~8 KB as text each; small chunks keep every statement modest.
_CHUNK = 100


@dataclass(frozen=True)
class JobRecord:
    origin: str
    origin_id: str
    source: str
    title: str
    company: str
    url: str
    location: str | None
    remote: bool | None
    seniority: str | None
    description: str
    requirements: str | None
    responsibilities: str | None
    content_hash: str | None
    embed_text: str
    embed_text_sha: str
    prod_embedding: list[float] | None


@dataclass(frozen=True)
class LabelRecord:
    origin_id: str
    grade: int
    source: str
    rationale: str | None = None


@dataclass
class TopicPayload:
    """Everything a topic is made of. Membership is every job in `jobs`."""

    tier: str
    name: str
    profile_snapshot: dict
    query_inputs: dict
    jobs: list[JobRecord]
    labels: list[LabelRecord]
    builder: str
    notes: str | None = None


@dataclass
class PersistSummary:
    topic_id: UUID
    n_jobs_new: int
    n_jobs_reused: int
    n_labels: int


class TopicExists(Exception):
    """A topic with this name is already stored; topics are frozen, never overwritten."""


class TopicBuilder(Protocol):
    tier: str

    async def build(self, name: str, **opts) -> TopicPayload: ...


def job_uuid(origin: str, origin_id: str, embed_text_sha: str) -> UUID:
    """The document id: the same posting with the same embed text is the same row everywhere."""
    return uuid5(_NAMESPACE, f"{origin}\x1f{origin_id}\x1f{embed_text_sha}")


def _chunks[T](items: Sequence[T]) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), _CHUNK):
        yield items[start : start + _CHUNK]


def _job_row(job_id: UUID, job: JobRecord) -> dict:
    return {
        "id": job_id,
        "origin": job.origin,
        "origin_id": job.origin_id,
        "source": job.source,
        "title": job.title,
        "company": job.company,
        "url": job.url,
        "location": job.location,
        "remote": job.remote,
        "seniority": job.seniority,
        "description": job.description,
        "requirements": job.requirements,
        "responsibilities": job.responsibilities,
        "content_hash": job.content_hash,
        "embed_text": job.embed_text,
        "embed_text_sha": job.embed_text_sha,
        "prod_embedding": job.prod_embedding,
    }


def _resolve(payload: TopicPayload) -> tuple[dict[UUID, JobRecord], list[tuple[UUID, LabelRecord]]]:
    """Map the payload to document ids and validate its labels, before anything is written."""
    jobs: dict[UUID, JobRecord] = {}
    by_origin_id: dict[str, UUID] = {}
    for job in payload.jobs:
        job_id = job_uuid(job.origin, job.origin_id, job.embed_text_sha)
        if by_origin_id.setdefault(job.origin_id, job_id) != job_id:
            raise ValueError(
                f"origin_id {job.origin_id!r} appears twice with different embed texts; "
                "labels could not be attached to one of them"
            )
        jobs.setdefault(job_id, job)

    labels: list[tuple[UUID, LabelRecord]] = []
    for label in payload.labels:
        if label.origin_id not in by_origin_id:
            raise ValueError(f"label for origin_id {label.origin_id!r} matches no job in the topic")
        if label.source not in SOURCES or label.source == "llm_judge":
            raise ValueError(
                f"label source {label.source!r} cannot be persisted with a topic; "
                f"expected one of {tuple(s for s in SOURCES if s != 'llm_judge')}"
            )
        if not 0 <= label.grade <= 3:
            raise ValueError(f"grade {label.grade} for {label.origin_id!r} is outside 0..3")
        labels.append((by_origin_id[label.origin_id], label))
    return jobs, labels


async def persist_topic(session: AsyncSession, payload: TopicPayload) -> PersistSummary:
    """Store a topic, its documents, membership and labels in one transaction.

    Documents are shared: an id already present (same origin, origin_id and embed text) is
    reused, never rewritten. The topic goes in as `draft` and is `frozen` only once everything
    else is in, so a half-written topic can never be seen as complete.
    """
    try:
        return await _persist(session, payload)
    except Exception:
        await session.rollback()
        raise


async def _persist(session: AsyncSession, payload: TopicPayload) -> PersistSummary:
    if await session.scalar(select(EmbeddingTopic.id).where(EmbeddingTopic.name == payload.name)):
        raise TopicExists(f"topic {payload.name!r} already exists")
    jobs, labels = _resolve(payload)

    topic = EmbeddingTopic(
        id=uuid4(),
        tier=payload.tier,
        name=payload.name,
        status="draft",
        profile_snapshot=payload.profile_snapshot,
        query_inputs=payload.query_inputs,
        builder=payload.builder,
        notes=payload.notes,
    )
    session.add(topic)
    await session.flush()

    job_rows = [_job_row(job_id, job) for job_id, job in jobs.items()]
    n_new = 0
    for chunk in _chunks(job_rows):
        stmt = (
            pg_insert(EmbeddingJob)
            .values(list(chunk))
            .on_conflict_do_nothing(index_elements=[EmbeddingJob.id])
            .returning(EmbeddingJob.id)
        )
        n_new += len((await session.execute(stmt)).all())

    for chunk in _chunks([{"topic_id": topic.id, "job_id": job_id} for job_id in jobs]):
        await session.execute(core_insert(EmbeddingTopicJob), list(chunk))

    label_rows = [
        {
            "topic_id": topic.id,
            "job_id": job_id,
            "grade": label.grade,
            "source": label.source,
            "rationale": label.rationale,
        }
        for job_id, label in labels
    ]
    for chunk in _chunks(label_rows):
        await session.execute(core_insert(EmbeddingLabel), list(chunk))

    topic.status = "frozen"
    await session.commit()
    return PersistSummary(
        topic_id=topic.id,
        n_jobs_new=n_new,
        n_jobs_reused=len(job_rows) - n_new,
        n_labels=len(label_rows),
    )
