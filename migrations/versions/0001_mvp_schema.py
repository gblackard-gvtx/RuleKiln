"""Initial MVP schema migration."""

import os

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

ENABLE_PGVECTOR = os.getenv("ENABLE_PGVECTOR", "false").lower() == "true"


def upgrade() -> None:
    if ENABLE_PGVECTOR:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "distillation_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("task_name", sa.String(), nullable=False),
        sa.Column("task_mode", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=True),
        sa.Column("request_json", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "cases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column("task_mode", sa.String(), nullable=False),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column("input_json", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("expected_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("expected_text", sa.Text(), nullable=True),
        sa.Column(
            "evaluation_json",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "metadata",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("weight", sa.Double(), nullable=False, server_default="1.0"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "micro_rules",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("condition", sa.Text(), nullable=False),
        sa.Column("expected_outcome", sa.Text(), nullable=False),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("rationale_summary", sa.Text(), nullable=True),
        sa.Column("rule_type", sa.String(), nullable=False, server_default="decision"),
        sa.Column(
            "positive_cues", sa.dialects.postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "negative_cues", sa.dialects.postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "rule_clusters",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=True),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("rule_ids", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "synthesized_rules",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("applies_when", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("outcome_conditions", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column(
            "tie_breakers", sa.dialects.postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("source_case_ids", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("source_micro_rule_ids", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("task_name", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("prompt_hash", sa.String(), nullable=False),
        sa.Column("mlflow_prompt_uri", sa.String(), nullable=True),
        sa.Column("is_selected", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column(
            "prompt_version_id",
            sa.String(),
            sa.ForeignKey("prompt_versions.id"),
            nullable=True,
        ),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column("accuracy", sa.Double(), nullable=True),
        sa.Column("macro_f1", sa.Double(), nullable=True),
        sa.Column("weighted_case_score", sa.Double(), nullable=True),
        sa.Column("per_outcome_precision", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("per_outcome_recall", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("malformed_output_rate", sa.Double(), nullable=True),
        sa.Column("confusion_matrix", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "stage_markers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey("distillation_jobs.id"), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=True),
        sa.Column("artifact_type", sa.String(), nullable=True),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "stage", "strategy", "artifact_type", name="uq_stage_marker"),
    )


def downgrade() -> None:
    op.drop_table("stage_markers")
    op.drop_table("eval_runs")
    op.drop_table("prompt_versions")
    op.drop_table("synthesized_rules")
    op.drop_table("rule_clusters")
    op.drop_table("micro_rules")
    op.drop_table("cases")
    op.drop_table("distillation_jobs")
