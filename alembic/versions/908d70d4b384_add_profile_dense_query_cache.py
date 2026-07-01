"""add_profile_dense_query_cache

Revision ID: 908d70d4b384
Revises: 76d5a8485cfe
Create Date: 2026-06-30 09:04:54.119820

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "908d70d4b384"
down_revision: str | Sequence[str] | None = "76d5a8485cfe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("profile", sa.Column("dense_query_cache", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("profile", "dense_query_cache")
