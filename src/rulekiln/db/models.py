"""SQLAlchemy ORM models for all MVP tables."""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class DistillationJob(Base):
    __tablename__ = "distillation_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    task_name: Mapped[str] = mapped_column(String, nullable=False)
    task_mode: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="created")
    stage: Mapped[str | None] = mapped_column(String, nullable=True)
    selected_strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    request_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Queue fields ──────────────────────────────────────────────────────
    queue_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    locked_by: Mapped[str | None] = mapped_column(String, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Token / cost summary fields ───────────────────────────────────────
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_total_cost_usd: Mapped[float | None] = mapped_column(
        Numeric(precision=12, scale=6), nullable=True
    )
    teacher_cost_usd: Mapped[float | None] = mapped_column(
        Numeric(precision=12, scale=6), nullable=True
    )
    student_cost_usd: Mapped[float | None] = mapped_column(
        Numeric(precision=12, scale=6), nullable=True
    )
    embedding_cost_usd: Mapped[float | None] = mapped_column(
        Numeric(precision=12, scale=6), nullable=True
    )
    judge_cost_usd: Mapped[float | None] = mapped_column(
        Numeric(precision=12, scale=6), nullable=True
    )

    cases: Mapped[list["Case"]] = relationship("Case", back_populates="job")
    micro_rules: Mapped[list["MicroRule"]] = relationship("MicroRule", back_populates="job")
    rule_clusters: Mapped[list["RuleCluster"]] = relationship("RuleCluster", back_populates="job")
    synthesized_rules: Mapped[list["SynthesizedRule"]] = relationship(
        "SynthesizedRule", back_populates="job"
    )
    prompt_versions: Mapped[list["PromptVersion"]] = relationship(
        "PromptVersion", back_populates="job"
    )
    eval_runs: Mapped[list["EvalRun"]] = relationship("EvalRun", back_populates="job")
    stage_markers: Mapped[list["StageMarker"]] = relationship("StageMarker", back_populates="job")
    model_call_events: Mapped[list["ModelCallEvent"]] = relationship(
        "ModelCallEvent", back_populates="job"
    )


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    task_mode: Mapped[str] = mapped_column(String, nullable=False)
    split: Mapped[str] = mapped_column(String, nullable=False)
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]
    expected_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # pyright: ignore[reportArgumentType]
    expected_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # pyright: ignore[reportArgumentType]
    metadata_json: Mapped[dict] = mapped_column(  # pyright: ignore[reportArgumentType]
        "metadata", JSON, nullable=False, default=dict
    )
    weight: Mapped[float] = mapped_column(Double, nullable=False, default=1.0)

    job: Mapped[DistillationJob] = relationship("DistillationJob", back_populates="cases")


class MicroRule(Base):
    __tablename__ = "micro_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    case_id: Mapped[str] = mapped_column(String, ForeignKey("cases.id"), nullable=False)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    condition: Mapped[str] = mapped_column(Text, nullable=False)
    expected_outcome: Mapped[str] = mapped_column(Text, nullable=False)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    rationale_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_type: Mapped[str] = mapped_column(String, nullable=False, default="decision")
    positive_cues: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # pyright: ignore[reportArgumentType]
    negative_cues: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # pyright: ignore[reportArgumentType]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[DistillationJob] = relationship("DistillationJob", back_populates="micro_rules")


class RuleCluster(Base):
    __tablename__ = "rule_clusters"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    algorithm: Mapped[str] = mapped_column(String, nullable=False)
    rule_ids: Mapped[list] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]
    cluster_metadata: Mapped[dict] = mapped_column(  # pyright: ignore[reportArgumentType]
        "metadata", JSON, nullable=False, default=dict
    )

    job: Mapped[DistillationJob] = relationship("DistillationJob", back_populates="rule_clusters")


class SynthesizedRule(Base):
    __tablename__ = "synthesized_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    rule_type: Mapped[str] = mapped_column(String, nullable=False, default="decision")
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    applies_when: Mapped[list] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]
    outcome_conditions: Mapped[dict] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]
    tie_breakers: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # pyright: ignore[reportArgumentType]
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    source_case_ids: Mapped[list] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]
    source_micro_rule_ids: Mapped[list] = mapped_column(JSON, nullable=False)  # pyright: ignore[reportArgumentType]

    # ── Conflict fields ───────────────────────────────────────────────────
    has_conflicts: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    conflict_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    conflicting_micro_rule_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # pyright: ignore[reportArgumentType]

    # ── Pruning / support fields ──────────────────────────────────────────
    support_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    support_ratio: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    golden_case_backed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    estimated_token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_pruned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pruning_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    job: Mapped[DistillationJob] = relationship(
        "DistillationJob", back_populates="synthesized_rules"
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    task_name: Mapped[str] = mapped_column(String, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String, nullable=False)
    mlflow_prompt_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    is_selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[DistillationJob] = relationship("DistillationJob", back_populates="prompt_versions")
    eval_runs: Mapped[list["EvalRun"]] = relationship("EvalRun", back_populates="prompt_version")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    prompt_version_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("prompt_versions.id"), nullable=True
    )
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    split: Mapped[str] = mapped_column(String, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Double, nullable=True)
    macro_f1: Mapped[float | None] = mapped_column(Double, nullable=True)
    weighted_case_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    per_outcome_precision: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # pyright: ignore[reportArgumentType]
    per_outcome_recall: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # pyright: ignore[reportArgumentType]
    malformed_output_rate: Mapped[float | None] = mapped_column(Double, nullable=True)
    confusion_matrix: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # pyright: ignore[reportArgumentType]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[DistillationJob] = relationship("DistillationJob", back_populates="eval_runs")
    prompt_version: Mapped[PromptVersion | None] = relationship(
        "PromptVersion", back_populates="eval_runs"
    )


class StageMarker(Base):
    """Durable stage completion marker for idempotent resume semantics."""

    __tablename__ = "stage_markers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    artifact_type: Mapped[str | None] = mapped_column(String, nullable=True)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[DistillationJob] = relationship("DistillationJob", back_populates="stage_markers")


class ModelCallEvent(Base):
    """Persisted record of a single model API call with token usage and cost."""

    __tablename__ = "model_call_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("distillation_jobs.id"), nullable=False)

    stage: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    provider_profile: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)

    student_id: Mapped[str | None] = mapped_column(String, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    case_id: Mapped[str | None] = mapped_column(String, nullable=True)

    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    input_cost_usd: Mapped[float] = mapped_column(
        Numeric(precision=12, scale=6), nullable=False, default=0
    )
    output_cost_usd: Mapped[float] = mapped_column(
        Numeric(precision=12, scale=6), nullable=False, default=0
    )
    total_cost_usd: Mapped[float] = mapped_column(
        Numeric(precision=12, scale=6), nullable=False, default=0
    )
    cost_estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    pricing_source: Mapped[str | None] = mapped_column(String, nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[DistillationJob] = relationship(
        "DistillationJob", back_populates="model_call_events"
    )
