"""add fit_judgments cache table

Revision ID: 30aa3a142086
Revises: 0c80cdb46d82
Create Date: 2026-08-08 11:23:39.162428

Autogenerate additionally proposed dropping `jobs_bm25`, `jobs_embedding_idx`, and
`profile_cv_embedding_idx`. Those are created by raw `op.execute` in earlier
revisions, so SQLAlchemy's metadata cannot see them and reports them as removed.
Dropping them would disable BM25 and vector retrieval outright — removed by hand.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "30aa3a142086"
down_revision: str | Sequence[str] | None = "0c80cdb46d82"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fit_judgments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("profile_id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False),
        sa.Column("judgment", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["profile_id"], ["profile.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id",
            "job_id",
            "content_hash",
            "model",
            "prompt_version",
            name="uq_fit_judgments_cache_key",
        ),
    )
    op.create_index(op.f("ix_fit_judgments_job_id"), "fit_judgments", ["job_id"], unique=False)
    op.create_index(
        op.f("ix_fit_judgments_profile_id"), "fit_judgments", ["profile_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_fit_judgments_profile_id"), table_name="fit_judgments")
    op.drop_index(op.f("ix_fit_judgments_job_id"), table_name="fit_judgments")
    op.drop_table("fit_judgments")
