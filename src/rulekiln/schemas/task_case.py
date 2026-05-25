"""Task and case domain schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRoute(BaseModel):
    """A provider profile + model pair for a specific role."""

    provider_profile: str
    model: str
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    max_concurrency: int | None = None


TaskMode = Literal[
    "classification",
    "summarization",
    "extraction",
    "rubric_review",
    "routing",
    "tool_use",
    "freeform_generation",
    "agent_behavior",
]


class BaselinePromptPolicy(BaseModel):
    """Controls which sections are included in the compiled baseline prompt."""

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        extra="ignore",
    )

    compiler: str = "default_baseline_v1"
    include_role: bool = True
    include_task_description: bool = True
    include_input_template: bool = True
    include_output_schema: bool = True
    include_allowed_values: bool = True
    include_prompt_scaffold: bool = True
    include_input_boundary: bool = True
    include_distilled_rules: bool = False


class RuleKilnTask(BaseModel):
    """Reusable task definition."""

    schema_version: Literal["rulekiln.task.v1"] = "rulekiln.task.v1"
    task_id: str
    task_name: str
    task_mode: TaskMode
    description: str
    input_template: str
    output_schema: dict[str, Any] = Field(default_factory=dict)
    prompt_scaffold: dict[str, Any] = Field(default_factory=dict)
    baseline_prompt_policy: BaselinePromptPolicy = Field(default_factory=BaselinePromptPolicy)
    allowed_evaluation_methods: list[str] = Field(default_factory=list)
    provider_model_defaults: dict[
        Literal["teacher", "student", "embedding", "judge"],
        ModelRoute,
    ] = Field(default_factory=dict)
    quality_gates: dict[str, Any] = Field(default_factory=dict)

    # ── Rule pruning budget ───────────────────────────────────────────────
    max_rules: int = 40
    max_prompt_tokens: int = 8000
    min_rule_support_count: int = 2
    preserve_golden_rules: bool = True


AssertionType = Literal[
    "must_include",
    "must_not_include",
    "must_equal",
    "must_match_regex",
    "json_schema",
    "semantic_match",
    "llm_judge",
]


class EvaluationAssertion(BaseModel):
    type: AssertionType
    path: str | None = None
    value: Any = None
    weight: float = 1.0


class RubricCriterion(BaseModel):
    name: str
    description: str
    weight: float = 1.0


class EvaluationSpec(BaseModel):
    primary_metric: str | None = None
    rubric: list[RubricCriterion] = Field(default_factory=list)
    assertions: list[EvaluationAssertion] = Field(default_factory=list)


class RuleKilnCase(BaseModel):
    """A single training or evaluation case."""

    schema_version: Literal["rulekiln.case.v1"] = "rulekiln.case.v1"
    id: str
    split: Literal["train", "validation", "test", "golden"] = "train"
    task_mode: TaskMode
    input: dict[str, Any]
    expected: dict[str, Any] | str | None = None
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
