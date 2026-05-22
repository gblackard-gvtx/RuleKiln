"""Add selected_strategy column to distillation_jobs."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "distillation_jobs",
        sa.Column("selected_strategy", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("distillation_jobs", "selected_strategy")
