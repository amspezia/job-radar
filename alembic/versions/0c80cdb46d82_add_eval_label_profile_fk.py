"""add_eval_label_profile_fk

Revision ID: 0c80cdb46d82
Revises: c7a1d0e2b435
Create Date: 2026-06-30 11:15:22.666251

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0c80cdb46d82"
down_revision: str | Sequence[str] | None = "c7a1d0e2b435"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "eval_labels",
        sa.Column("profile_id", postgresql.UUID(as_uuid=True), nullable=False),
    )
    op.create_foreign_key(
        "fk_eval_labels_profile_id",
        "eval_labels",
        "profile",
        ["profile_id"],
        ["id"],
    )
    op.create_index("ix_eval_labels_profile_id", "eval_labels", ["profile_id"])


def downgrade() -> None:
    op.drop_index("ix_eval_labels_profile_id", table_name="eval_labels")
    op.drop_constraint("fk_eval_labels_profile_id", "eval_labels", type_="foreignkey")
    op.drop_column("eval_labels", "profile_id")
