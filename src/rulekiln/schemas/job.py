"""Job request and response schemas."""

from pydantic import BaseModel, model_validator

from rulekiln.schemas.task_case import ModelRoute, RuleKilnCase, RuleKilnTask

# Legacy top-level fields that strict mode rejects
_FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {"task_name", "task_description", "labels", "examples"}
)


class DistillationRequest(BaseModel):
    """Strict canonical envelope for a distillation job submission."""

    task: RuleKilnTask
    cases: list[RuleKilnCase]
    teacher: ModelRoute
    student: ModelRoute
    embedding: ModelRoute
    judge: ModelRoute | None = None
    baseline_prompt: str | None = None
    metric: str | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, values: dict) -> dict:  # pyright: ignore[reportUnknownParameterType]
        found = _FORBIDDEN_FIELDS & set(values.keys())
        if found:
            raise ValueError(
                f"Legacy top-level fields are not accepted: {sorted(found)}. "
                "Use the canonical envelope: task, cases, teacher, student, embedding."
            )
        return values  # pyright: ignore[reportReturnType]


class JobProgress(BaseModel):
    completed: int
    total: int


class JobUsageSummary(BaseModel):
    """Token usage and cost summary for a distillation job."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_total_cost_usd: float | None = None
    teacher_cost_usd: float | None = None
    student_cost_usd: float | None = None
    embedding_cost_usd: float | None = None
    judge_cost_usd: float | None = None
    has_estimated_usage: bool = False
    total_model_calls: int = 0


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    progress: JobProgress | None = None
    error_message: str | None = None
    usage: JobUsageSummary | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
