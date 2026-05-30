"""Add model_call_events table and usage summary columns to distillation_jobs.

Revision ID: 0004_add_model_call_events
Revises: 0003
Create Date: 2025-01-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_add_model_call_events"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add usage summary columns to distillation_jobs
    op.add_column(
        "distillation_jobs",
        sa.Column("total_input_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("total_output_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("estimated_total_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("teacher_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("student_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("embedding_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("judge_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
    )

    # Create model_call_events table
    op.create_table(
        "model_call_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("provider_profile", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("student_id", sa.String(), nullable=True),
        sa.Column("strategy", sa.String(), nullable=True),
        sa.Column("case_id", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("usage_estimated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "input_cost_usd",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "output_cost_usd",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "total_cost_usd",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column("cost_estimated", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("pricing_source", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_type", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["distillation_jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_call_events_job_id", "model_call_events", ["job_id"], unique=False)
    op.create_index("ix_model_call_events_stage", "model_call_events", ["stage"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_model_call_events_stage", table_name="model_call_events")
    op.drop_index("ix_model_call_events_job_id", table_name="model_call_events")
    op.drop_table("model_call_events")

    op.drop_column("distillation_jobs", "judge_cost_usd")
    op.drop_column("distillation_jobs", "embedding_cost_usd")
    op.drop_column("distillation_jobs", "student_cost_usd")
    op.drop_column("distillation_jobs", "teacher_cost_usd")
    op.drop_column("distillation_jobs", "estimated_total_cost_usd")
    op.drop_column("distillation_jobs", "total_tokens")
    op.drop_column("distillation_jobs", "total_output_tokens")
    op.drop_column("distillation_jobs", "total_input_tokens")
