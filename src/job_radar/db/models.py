from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, Computed, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
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
    # Mirrors the GENERATED ALWAYS AS clause from the migration. Computed()
    # tells the ORM this column is database-maintained, so it's excluded from
    # every INSERT/UPDATE the unit-of-work issues — without it, SQLAlchemy
    # sends an explicit NULL for any unset attribute, which Postgres rejects
    # outright for a generated column.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('simple', coalesce(title, '')), 'A') || "
            "setweight(to_tsvector('simple', coalesce(description, '')), 'B')",
            persisted=True,
        ),
        deferred=True,
    )


class Profile(Base):
    __tablename__ = "profile"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
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
    remote_required: Mapped[bool] = mapped_column(Boolean)


class EvalLabel(Base):
    __tablename__ = "eval_labels"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("jobs.id"), index=True)
    label: Mapped[str] = mapped_column(String(50))
    labeled_by: Mapped[str] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text)
