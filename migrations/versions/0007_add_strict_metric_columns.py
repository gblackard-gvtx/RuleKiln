"""Add metric_semantics_version and valid_output_* columns to eval_runs."""

import sqlalchemy as sa
from alembic import op

revision = "0007_add_strict_metric_columns"
down_revision = "0006_add_batch_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_runs",
        sa.Column("metric_semantics_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "eval_runs",
        sa.Column("valid_output_accuracy", sa.Double(), nullable=True),
    )
    op.add_column(
        "eval_runs",
        sa.Column("valid_output_macro_f1", sa.Double(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("eval_runs", "valid_output_macro_f1")
    op.drop_column("eval_runs", "valid_output_accuracy")
    op.drop_column("eval_runs", "metric_semantics_version")
