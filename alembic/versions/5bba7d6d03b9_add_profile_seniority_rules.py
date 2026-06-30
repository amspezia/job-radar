"""add profile.seniority_rules

Revision ID: 5bba7d6d03b9
Revises: 8def70e51bd8
Create Date: 2026-06-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5bba7d6d03b9"
down_revision: str | Sequence[str] | None = "8def70e51bd8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("profile", sa.Column("seniority_rules", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("profile", "seniority_rules")
