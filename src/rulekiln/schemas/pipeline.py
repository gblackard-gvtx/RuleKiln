"""Pipeline domain schemas: rule extraction, clustering, synthesis, evaluation, quality gates."""

from typing import Literal

from pydantic import BaseModel, Field

# ── Rule extraction ──────────────────────────────────────────────────────────


class MicroRuleSchema(BaseModel):
    """A single rule extracted from a teacher-model case response."""

    topic: str
    condition: str
    expected_outcome: str
    output_path: str | None = None
    rationale_summary: str | None = None
    rule_type: str = "decision"
    positive_cues: list[str] = Field(default_factory=list)
    negative_cues: list[str] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    """Structured output from the rule-extraction agent."""

    rules: list[MicroRuleSchema] = Field(default_factory=list)
    reasoning: str | None = None


# ── Synthesis ────────────────────────────────────────────────────────────────


class OutcomeCondition(BaseModel):
    outcome: str
    when: list[str] = Field(default_factory=list)
    confidence: str = "high"


class SynthesizedRuleSchema(BaseModel):
    """A synthesized rule derived from a cluster of micro-rules."""

    id: str = ""
    rule_type: str = "decision"
    topic: str
    applies_when: list[str] = Field(default_factory=list)
    outcome_conditions: dict[str, OutcomeCondition] = Field(default_factory=dict)
    tie_breakers: list[str] = Field(default_factory=list)
    priority: int = 100
    source_case_ids: list[str] = Field(default_factory=list)
    source_micro_rule_ids: list[str] = Field(default_factory=list)

    # ── Conflict fields ────────────────────────────────────────────────
    has_conflicts: bool = False
    conflict_summary: str | None = None
    conflicting_micro_rule_ids: list[str] = Field(default_factory=list)

    # ── Pruning / support metadata ────────────────────────────────────────
    support_count: int = 0
    support_ratio: float = 0.0
    golden_case_backed: bool = False
    estimated_token_count: int = 0


class SynthesisOutput(BaseModel):
    """Structured output from the rule-synthesis agent."""

    rules: list[SynthesizedRuleSchema] = Field(default_factory=list)
    reasoning: str | None = None


# ── Conflict review ──────────────────────────────────────────────────────


class RuleConflictReview(BaseModel):
    """Conflict review result for a single synthesized rule."""

    synthesized_rule_id: str
    has_conflicts: bool
    conflict_summary: str | None = None
    conflicting_micro_rule_ids: list[str] = Field(default_factory=list)
    resolution: Literal["keep", "modify", "split", "discard"]
    resolved_rules: list[SynthesizedRuleSchema] = Field(default_factory=list)


# ── Clustering ───────────────────────────────────────────────────────────────


class RuleClusterSchema(BaseModel):
    """A cluster of micro-rule IDs produced by a clustering algorithm."""

    strategy: str
    topic: str | None = None
    algorithm: str
    rule_ids: list[str]
    cluster_metadata: dict[str, str | int | float] = Field(default_factory=dict)


# ── Evaluation ───────────────────────────────────────────────────────────────


class CaseEvalResult(BaseModel):
    """Evaluation result for a single case."""

    case_id: str
    score: float
    passed: bool
    malformed: bool = False
    assertion_scores: dict[str, float] = Field(default_factory=dict)
    actual_output: dict[str, str | int | float | bool | None] | str | None = None
    error: str | None = None


class EvalResult(BaseModel):
    """Aggregate evaluation result for a prompt version on a split."""

    strategy: str
    model: str
    split: str
    accuracy: float | None = None
    macro_f1: float | None = None
    weighted_case_score: float | None = None
    malformed_output_rate: float = 0.0
    per_outcome_precision: dict[str, float] = Field(default_factory=dict)
    per_outcome_recall: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    case_results: list[CaseEvalResult] = Field(default_factory=list)
    violated_rule_counts: dict[str, int] = Field(default_factory=dict)
    failed_assertion_path_counts: dict[str, int] = Field(default_factory=dict)
    failures: list["CaseEvaluationFailure"] = Field(default_factory=list)


# ── Eval-to-rule failure mapping ───────────────────────────────────────────


class CaseEvaluationFailure(BaseModel):
    """Granular failure record with rule mapping."""

    case_id: str
    split: str
    failure_class: Literal["fixed", "broken", "unchanged_wrong"]
    matched_rule_ids: list[str] = Field(default_factory=list)
    violated_rule_ids: list[str] = Field(default_factory=list)
    failed_assertion_paths: list[str] = Field(default_factory=list)
    failed_assertion_types: list[str] = Field(default_factory=list)
    explanation: str | None = None


# ── Quality gates ─────────────────────────────────────────────────────────────


class QualityGateResult(BaseModel):
    """Result of a quality gate check for one strategy."""

    strategy: str
    passed: bool
    metric_delta: float | None = None
    regression_rate: float = 0.0
    golden_failures: int = 0
    malformed_output_rate: float = 0.0
    prompt_tokens: int = 0
    violations: list[str] = Field(default_factory=list)


# ── Strategy comparison ───────────────────────────────────────────────────────


class StrategyComparison(BaseModel):
    """Full comparison across strategies after evaluation and gate checks."""

    baseline_eval: EvalResult | None = None
    dbscan_eval: EvalResult | None = None
    hdbscan_eval: EvalResult | None = None
    dbscan_gate: QualityGateResult | None = None
    hdbscan_gate: QualityGateResult | None = None
    selected_strategy: str | None = None
    selection_reason: str | None = None
    evaluation_split_warning: str | None = None
