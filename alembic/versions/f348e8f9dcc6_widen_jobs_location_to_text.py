"""widen jobs.location to text

Revision ID: f348e8f9dcc6
Revises: 1c79f1288961
Create Date: 2026-06-28 10:47:50.352603

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f348e8f9dcc6"
down_revision: str | Sequence[str] | None = "1c79f1288961"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "jobs",
        "location",
        type_=sa.Text(),
        existing_type=sa.String(length=255),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "jobs",
        "location",
        type_=sa.String(length=255),
        existing_type=sa.Text(),
        existing_nullable=True,
    )
