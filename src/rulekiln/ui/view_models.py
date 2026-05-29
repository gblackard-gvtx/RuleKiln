"""Pydantic view models for the operator UI — no bare dicts."""

from datetime import datetime

from pydantic import BaseModel, Field


class JobListItemView(BaseModel):
    """Summary row shown in the job list dashboard."""

    job_id: str
    task_name: str
    task_mode: str
    status: str
    stage: str | None
    selected_strategy: str | None
    primary_metric_delta: float | None
    created_at: datetime
    detail_url: str


class JobDetailView(BaseModel):
    """Full detail for a single distillation job."""

    job_id: str
    task_name: str
    task_mode: str
    status: str
    stage: str | None
    progress_completed: int | None
    progress_total: int | None
    selected_strategy: str | None
    quality_gates_passed: bool | None
    mlflow_run_id: str | None
    mlflow_run_url: str | None
    error_message: str | None
    total_cases: int | None = None
    train_cases: int | None = None
    validation_cases: int | None = None
    test_cases: int | None = None
    golden_cases: int | None = None
    teacher_extraction_completed: int | None = None
    teacher_extraction_total: int | None = None
    student_eval_split: str | None = None
    student_eval_total: int | None = None
    student_baseline_completed: int | None = None
    student_dbscan_completed: int | None = None
    student_hdbscan_completed: int | None = None
    total_model_calls: int | None = None
    teacher_model_calls: int | None = None
    student_model_calls: int | None = None
    embedding_model_calls: int | None = None
    judge_model_calls: int | None = None
    micro_rules_count: int | None = None
    synthesized_rules_count: int | None = None
    selected_rules_count: int | None = None


class ResultsSummaryView(BaseModel):
    """Metric comparison across strategies for a completed job."""

    job_id: str
    primary_metric: str
    baseline_score: float | None
    dbscan_score: float | None
    hdbscan_score: float | None
    selected_score: float | None
    selected_strategy: str | None
    metric_delta: float | None
    golden_failures: int | None
    malformed_output_rate: float | None
    prompt_token_count: int | None
    fixed_count: int | None
    broken_count: int | None
    quality_gates_passed: bool | None
    best_strategy: str | None = None
    baseline_macro_f1: float | None = None
    best_macro_f1: float | None = None
    macro_f1_delta: float | None = None
    macro_f1_relative_lift_percent: float | None = None
    accuracy_lift_percentage_points: float | None = None
    best_malformed_output_rate: float | None = None
    # Token / cost fields
    estimated_total_cost_usd: float | None = None
    teacher_cost_usd: float | None = None
    student_cost_usd: float | None = None
    embedding_cost_usd: float | None = None
    judge_cost_usd: float | None = None
    total_model_calls: int | None = None
    total_tokens: int | None = None
    has_estimated_usage: bool = False


class ProviderRouteView(BaseModel):
    """Provider profile + model pair for display."""

    profile_name: str
    model_id: str
    supports_chat: bool
    supports_embeddings: bool


class PreviewView(BaseModel):
    """Parsed and validated job preview before final submission."""

    task_id: str
    task_name: str
    task_mode: str
    case_count: int
    train_count: int
    validation_count: int
    test_count: int
    golden_count: int
    evaluation_methods: list[str]
    output_schema_present: bool
    provider_routes: list[ProviderRouteView]
    estimated_teacher_calls: int
    estimated_student_eval_calls: int
    estimated_embedding_calls: int
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ArtifactFileView(BaseModel):
    """A single downloadable artifact file."""

    filename: str
    relative_path: str
    download_url: str
    content_type: str


class ArtifactsView(BaseModel):
    """Manifest of all artifact files for a job."""

    job_id: str
    files: list[ArtifactFileView]
