"""add profile.years_experience

Revision ID: 8def70e51bd8
Revises: 1beb38bb5629
Create Date: 2026-06-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8def70e51bd8"
down_revision: str | Sequence[str] | None = "1beb38bb5629"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("profile", sa.Column("years_experience", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("profile", "years_experience")
