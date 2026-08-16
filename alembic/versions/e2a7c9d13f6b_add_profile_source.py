"""add_profile_source

Revision ID: e2a7c9d13f6b
Revises: 30aa3a142086
Create Date: 2026-08-16 19:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e2a7c9d13f6b"
down_revision: str | Sequence[str] | None = "30aa3a142086"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "profile",
        sa.Column("source", sa.String(length=50), nullable=False, server_default="real"),
    )


def downgrade() -> None:
    op.drop_column("profile", "source")
