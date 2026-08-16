from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from job_radar.db.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(255))
    ingested_via: Mapped[str] = mapped_column(String(255))
    source_id: Mapped[str | None] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(String(2048), unique=True)
    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    # Structured fields extracted from description at ingest by an LLM pass.
    # These are what the BM25 index covers — not the raw description — so each
    # field has its own length normalization and company boilerplate is excluded.
    # NULL when extraction fails; those postings still reach the vector arms.
    requirements: Mapped[str | None] = mapped_column(Text)
    responsibilities: Mapped[str | None] = mapped_column(Text)
    # Canonical seniority level normalized from the title at ingest; NULL when
    # the posting states no recognizable level ("unknown"). The single source of
    # truth for level — both retrieval filtering and fit scoring read it.
    seniority: Mapped[str | None] = mapped_column(String(50))
    salary_min: Mapped[int | None] = mapped_column(Integer)
    salary_max: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str | None] = mapped_column(String(10))
    location: Mapped[str | None] = mapped_column(Text)
    remote: Mapped[bool] = mapped_column(Boolean)
    job_type: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)


class Profile(Base):
    __tablename__ = "profile"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    # "real" (the single actual candidate) or "synthetic" (an eval persona, see
    # eval/inject_synthetic.py) — every unscoped "find the profile" lookup (loader.py,
    # fit/pipeline.py) must filter on this, since synthetic personas live in this same
    # table and an unscoped `.first()` can otherwise silently return one of them.
    source: Mapped[str] = mapped_column(String(50), nullable=False, server_default="real")
    full_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255))
    links: Mapped[dict] = mapped_column(JSON)
    work_history: Mapped[dict] = mapped_column(JSON)
    cv_text: Mapped[str] = mapped_column(Text)
    cv_embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    target_titles: Mapped[dict] = mapped_column(JSON)
    seniority: Mapped[str] = mapped_column(String(255))
    years_experience: Mapped[float | None] = mapped_column(Float)
    domains_keywords: Mapped[dict] = mapped_column(JSON)
    salary_floor: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str | None] = mapped_column(String(10))
    location_rules: Mapped[dict] = mapped_column(JSON)
    seniority_rules: Mapped[dict | None] = mapped_column(JSON)
    remote_required: Mapped[bool] = mapped_column(Boolean)
    dense_query_cache: Mapped[str | None] = mapped_column(Text)


class FitJudgmentCache(Base):
    """A persisted LLM fit *judgment*, so a re-run analyzes only what changed.

    Analyzing one posting costs ~800 LLM output tokens at ~9 tok/s — the dominant
    cost of a fit run. Almost none of that work differs between consecutive runs,
    so the model's output is cached and invalidated by key rather than recomputed.

    **Only the LLM's grounded judgment is stored — never the score.** The score,
    verdict, and gates are recomputed by `score_fit` on every read. That is what
    keeps the cache correct: those depend on run-time inputs the judgment does not
    (`--level` overrides) and on constants that change independently (`_WEIGHTS`,
    `_BANDS`). Caching the number would serve a stale score whenever either moved;
    recomputing it is free, since `score_fit` is pure arithmetic.

    The five-column unique key IS the invalidation policy:
      profile_id     — judgments are relative to a candidate,
      job_id         — and to a posting,
      content_hash   — copied from the Job; an edited posting re-analyzes,
      model          — a different model is a different judge,
      prompt_version — a different prompt/schema is a different question.
    Dropping either of the last two would silently serve judgments produced
    under instructions the current code no longer uses.
    """

    __tablename__ = "fit_judgments"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "job_id",
            "content_hash",
            "model",
            "prompt_version",
            name="uq_fit_judgments_cache_key",
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("profile.id"), index=True
    )
    job_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("jobs.id"), index=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[int] = mapped_column(Integer)
    judgment: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EvalLabel(Base):
    __tablename__ = "eval_labels"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("profile.id"), index=True
    )
    job_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("jobs.id"), index=True)
    label: Mapped[str] = mapped_column(String(50))
    labeled_by: Mapped[str] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text)
