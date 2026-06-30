"""replace tsvector with bm25 paradedb

Revision ID: a3f9e2c1d504
Revises: 908d70d4b384
Create Date: 2026-06-30 09:15:00.000000

Replaces the ts_rank / tsvector / GIN lexical arm with a real BM25 index
via ParadeDB pg_search 0.24+. Requires the paradedb/paradedb Postgres image.

Title uses 'whitespace' tokenizer to preserve tech symbols (C++, C#, .NET).
Description uses 'default' tokenizer with Snowball English stemmer for
morphological recall (developer/developers, engineer/engineering).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a3f9e2c1d504"
down_revision: str | Sequence[str] | None = "908d70d4b384"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")

    op.execute("""
        CREATE INDEX jobs_bm25 ON jobs
        USING bm25 (id, title, description)
        WITH (
            key_field = 'id',
            text_fields = '{"title": {"tokenizer": {"type": "whitespace"}}, '
                '"description": {"tokenizer": {"type": "default", "stemmer": "English"}}}'
        )
    """)

    op.execute("DROP INDEX IF EXISTS ix_jobs_search_vector")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS search_vector")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS jobs_bm25")

    op.execute("""
        ALTER TABLE jobs ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
            setweight(to_tsvector('simple', coalesce(description, '')), 'B')
        ) STORED
    """)
    op.execute("CREATE INDEX ix_jobs_search_vector ON jobs USING GIN (search_vector)")
