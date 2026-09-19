"""Tables of the `embedding` eval family, all prefixed `embedding_`.

The schema contract is docs/plans/EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md §7.1. Open
vocabularies (`source`, `status`, `origin`) are plain TEXT validated in code so that adding a
value never needs a migration; `grade` is the one CHECK-constrained column.
"""

from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from eval.evals_db.base import EvalsBase


class EmbeddingJob(EvalsBase):
    """Immutable document snapshot. `id` is a deterministic uuid5 (tiers/base.py::job_uuid)."""

    __tablename__ = "embedding_job"
    __table_args__ = (
        UniqueConstraint("origin", "origin_id", "embed_text_sha", name="uq_embedding_job_identity"),
        Index("ix_embedding_job_embed_text_sha", "embed_text_sha"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    origin: Mapped[str] = mapped_column(Text)
    origin_id: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    company: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    remote: Mapped[bool | None] = mapped_column(Boolean)
    seniority: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    requirements: Mapped[str | None] = mapped_column(Text)
    responsibilities: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text)
    embed_text: Mapped[str] = mapped_column(Text)
    embed_text_sha: Mapped[str] = mapped_column(Text)
    prod_embedding: Mapped[np.ndarray | None] = mapped_column(Vector())


class EmbeddingTopic(EvalsBase):
    __tablename__ = "embedding_topic"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tier: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text, unique=True)
    status: Mapped[str] = mapped_column(Text)
    profile_snapshot: Mapped[dict] = mapped_column(JSONB)
    query_inputs: Mapped[dict] = mapped_column(JSONB)
    builder: Mapped[str] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingTopicJob(EvalsBase):
    __tablename__ = "embedding_topic_job"

    topic_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_topic.id"), primary_key=True)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_job.id"), primary_key=True)


class EmbeddingJudgeRun(EvalsBase):
    __tablename__ = "embedding_judge_run"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    topic_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_topic.id"))
    model: Mapped[str] = mapped_column(Text)
    model_digest: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    prompt_sha: Mapped[str] = mapped_column(Text)
    params: Mapped[dict] = mapped_column(JSONB)
    fewshot: Mapped[dict | list] = mapped_column(JSONB)
    n_labeled: Mapped[int | None] = mapped_column(Integer)
    n_failed: Mapped[int | None] = mapped_column(Integer)
    seconds: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingLabel(EvalsBase):
    """Append-only: a re-label adds a row. `id` orders rows (larger = newer)."""

    __tablename__ = "embedding_label"
    __table_args__ = (
        CheckConstraint("grade BETWEEN 0 AND 3", name="ck_embedding_label_grade"),
        Index("ix_embedding_label_topic_job", "topic_id", "job_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    topic_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_topic.id"))
    job_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_job.id"))
    grade: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(Text)
    judge_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("embedding_judge_run.id"))
    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingEmbedder(EvalsBase):
    __tablename__ = "embedding_embedder"

    fingerprint: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    digest: Mapped[str] = mapped_column(Text)
    quantization: Mapped[str | None] = mapped_column(Text)
    runtime_version: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingVector(EvalsBase):
    """The embedding cache, keyed by the exact string sent (prefix included)."""

    __tablename__ = "embedding_vector"

    fingerprint: Mapped[str] = mapped_column(
        ForeignKey("embedding_embedder.fingerprint"), primary_key=True
    )
    text_sha: Mapped[str] = mapped_column(Text, primary_key=True)
    vector: Mapped[np.ndarray] = mapped_column(Vector())
    n_tokens: Mapped[int | None] = mapped_column(Integer)
    truncated: Mapped[bool] = mapped_column(Boolean, server_default="false")


class EmbeddingRun(EvalsBase):
    __tablename__ = "embedding_run"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    topic_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_topic.id"))
    name: Mapped[str] = mapped_column(Text)
    embedder_fingerprint: Mapped[str | None] = mapped_column(
        ForeignKey("embedding_embedder.fingerprint")
    )
    method_fingerprint: Mapped[str] = mapped_column(Text)
    method_config: Mapped[dict] = mapped_column(JSONB)
    git_sha: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    docs_per_s: Mapped[float | None] = mapped_column(Float)
    n_truncated: Mapped[int | None] = mapped_column(Integer)
    seconds: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EmbeddingRunRanking(EvalsBase):
    __tablename__ = "embedding_run_ranking"
    __table_args__ = (Index("ix_embedding_run_ranking_run_rank", "run_id", "rank"),)

    run_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_run.id"), primary_key=True)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_job.id"), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)


class EmbeddingCheck(EvalsBase):
    __tablename__ = "embedding_check"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    topic_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_topic.id"))
    name: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool] = mapped_column(Boolean)
    detail: Mapped[dict] = mapped_column(JSONB)
    judge_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("embedding_judge_run.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
