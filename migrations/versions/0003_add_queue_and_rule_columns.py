"""Add queue columns to distillation_jobs and conflict/pruning columns to synthesized_rules."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Queue columns for distillation_jobs ──────────────────────────────
    op.add_column(
        "distillation_jobs",
        sa.Column("queue_status", sa.String(), nullable=False, server_default="pending"),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("locked_by", sa.String(), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    )
    op.add_column(
        "distillation_jobs",
        sa.Column(
            "next_run_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ── Queue claim indexes ───────────────────────────────────────────────
    op.create_index(
        "idx_distillation_jobs_queue_claim",
        "distillation_jobs",
        ["queue_status", "next_run_at", "created_at"],
    )
    op.create_index(
        "idx_distillation_jobs_lease",
        "distillation_jobs",
        ["queue_status", "lease_expires_at"],
    )

    # ── Conflict columns for synthesized_rules ────────────────────────────
    op.add_column(
        "synthesized_rules",
        sa.Column("rule_type", sa.String(), nullable=False, server_default="decision"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("has_conflicts", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("conflict_summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column(
            "conflicting_micro_rule_ids",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )

    # ── Pruning / support columns for synthesized_rules ───────────────────
    op.add_column(
        "synthesized_rules",
        sa.Column("support_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("support_ratio", sa.Double(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("golden_case_backed", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("estimated_token_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("is_pruned", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "synthesized_rules",
        sa.Column("pruning_reason", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("synthesized_rules", "pruning_reason")
    op.drop_column("synthesized_rules", "is_pruned")
    op.drop_column("synthesized_rules", "estimated_token_count")
    op.drop_column("synthesized_rules", "golden_case_backed")
    op.drop_column("synthesized_rules", "support_ratio")
    op.drop_column("synthesized_rules", "support_count")
    op.drop_column("synthesized_rules", "conflicting_micro_rule_ids")
    op.drop_column("synthesized_rules", "conflict_summary")
    op.drop_column("synthesized_rules", "has_conflicts")
    op.drop_column("synthesized_rules", "rule_type")

    op.drop_index("idx_distillation_jobs_lease", table_name="distillation_jobs")
    op.drop_index("idx_distillation_jobs_queue_claim", table_name="distillation_jobs")

    op.drop_column("distillation_jobs", "next_run_at")
    op.drop_column("distillation_jobs", "max_attempts")
    op.drop_column("distillation_jobs", "attempt_count")
    op.drop_column("distillation_jobs", "lease_expires_at")
    op.drop_column("distillation_jobs", "locked_at")
    op.drop_column("distillation_jobs", "locked_by")
    op.drop_column("distillation_jobs", "queue_status")
