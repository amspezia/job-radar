"""add jobs.seniority

Revision ID: 76d5a8485cfe
Revises: 5bba7d6d03b9
Create Date: 2026-06-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "76d5a8485cfe"
down_revision: str | Sequence[str] | None = "5bba7d6d03b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("jobs", sa.Column("seniority", sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("jobs", "seniority")
