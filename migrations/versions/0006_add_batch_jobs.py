import sqlalchemy as sa
from alembic import op

"""Add batch_jobs table for durable batch API submission state.

Revision ID intentionally kept <= 32 chars for alembic_version.version_num compatibility.
"""

revision = "0006_add_batch_jobs"
down_revision = "0005_eval_case_results"
branch_labels = None
depends_on = None




def upgrade() -> None:
    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("provider_batch_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="submitted"),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("succeeded_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("errored_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_file_id", sa.String(), nullable=True),
        sa.Column("output_file_id", sa.String(), nullable=True),
        sa.Column("error_file_id", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
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
            "stage",
            "strategy",
            "provider_batch_id",
            name="uq_batch_jobs_identity",
        ),
    )
    op.create_index(
        "ix_batch_jobs_lookup",
        "batch_jobs",
        ["job_id", "stage", "strategy", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_batch_jobs_lookup", table_name="batch_jobs")
    op.drop_table("batch_jobs")
