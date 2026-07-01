"""add extracted fields and rebuild bm25 index

Revision ID: c7a1d0e2b435
Revises: a3f9e2c1d504
Create Date: 2026-06-30 10:00:00.000000

Adds jobs.requirements and jobs.responsibilities, populated by an LLM
extraction pass at ingest time. The BM25 index is rebuilt to cover these
two extracted fields instead of the raw description, giving each field
its own independent length normalisation.

This eliminates the source bias caused by Greenhouse postings being 2x
longer than Himalayas postings on average: description is no longer
indexed for BM25 (it is still stored for fit analysis and display).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c7a1d0e2b435"
down_revision: str | Sequence[str] | None = "a3f9e2c1d504"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("requirements", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("responsibilities", sa.Text(), nullable=True))

    op.execute("DROP INDEX IF EXISTS jobs_bm25")
    op.execute(
        """
        CREATE INDEX jobs_bm25 ON jobs
        USING bm25 (id, title, requirements, responsibilities)
        WITH (
            key_field = 'id',
            text_fields = '{"title": {"tokenizer": {"type": "whitespace"}}, '
                '"requirements": {"tokenizer": {"type": "default", "stemmer": "English"}}, '
                '"responsibilities": {"tokenizer": {"type": "default", "stemmer": "English"}}}'
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS jobs_bm25")
    op.execute(
        """
        CREATE INDEX jobs_bm25 ON jobs
        USING bm25 (id, title, description)
        WITH (
            key_field = 'id',
            text_fields = '{"title": {"tokenizer": {"type": "whitespace"}}, '
                '"description": {"tokenizer": {"type": "default", "stemmer": "English"}}}'
        )
        """
    )

    op.drop_column("jobs", "requirements")
    op.drop_column("jobs", "responsibilities")
