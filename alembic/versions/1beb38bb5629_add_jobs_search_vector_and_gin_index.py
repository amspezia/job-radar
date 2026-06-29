"""add jobs search_vector and gin index

Revision ID: 1beb38bb5629
Revises: f348e8f9dcc6
Create Date: 2026-06-28 22:21:29.557164

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1beb38bb5629"
down_revision: str | Sequence[str] | None = "f348e8f9dcc6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# 'simple' (no stemming) is deliberate: 'english' mangles short tech tokens
# ("iOS" -> "io", "AWS" -> "aw"), which is worse for this corpus than losing
# stemming on generic words. Title outranks description (weight A vs B).
_SEARCH_VECTOR_EXPR = """
    setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(description, '')), 'B')
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"ALTER TABLE jobs ADD COLUMN search_vector tsvector "
        f"GENERATED ALWAYS AS ({_SEARCH_VECTOR_EXPR}) STORED"
    )
    op.execute("CREATE INDEX ix_jobs_search_vector ON jobs USING GIN (search_vector)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX ix_jobs_search_vector")
    op.execute("ALTER TABLE jobs DROP COLUMN search_vector")
