"""Add eval_case_results table and idempotency key support for model_call_events.

Revision ID intentionally kept <= 32 chars for alembic_version.version_num compatibility.
"""

from alembic import op
import sqlalchemy as sa

revision = "0005_eval_case_results"
down_revision = "0004_add_model_call_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_case_results",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("student_id", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("expected_json", sa.JSON(), nullable=True),
        sa.Column("actual_json", sa.JSON(), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.Column("assertion_scores", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("case_score", sa.Double(), nullable=False, server_default="0"),
        sa.Column("malformed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("invalid_label", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("error_type", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["distillation_jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id",
            "student_id",
            "strategy",
            "split",
            "case_id",
            name="uq_eval_case_results_job_student_strategy_split_case",
        ),
    )
    op.create_index(
        "ix_eval_case_results_job_strategy_split",
        "eval_case_results",
        ["job_id", "student_id", "strategy", "split"],
        unique=False,
    )

    op.add_column("model_call_events", sa.Column("idempotency_key", sa.String(), nullable=True))
    op.create_index(
        "ix_model_call_events_idempotency_key",
        "model_call_events",
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_model_call_events_idempotency_key", table_name="model_call_events")
    op.drop_column("model_call_events", "idempotency_key")

    op.drop_index("ix_eval_case_results_job_strategy_split", table_name="eval_case_results")
    op.drop_table("eval_case_results")
