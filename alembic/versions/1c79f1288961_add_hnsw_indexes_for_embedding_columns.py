"""add hnsw indexes for embedding columns

Revision ID: 1c79f1288961
Revises: 4a1001525d0a
Create Date: 2026-06-27 11:08:38.509945

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1c79f1288961"
down_revision: str | Sequence[str] | None = "4a1001525d0a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE INDEX ON jobs USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX ON profile USING hnsw (cv_embedding vector_cosine_ops)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS jobs_embedding_idx")
    op.execute("DROP INDEX IF EXISTS profile_cv_embedding_idx")
