"""Pipeline domain schemas: rule extraction, clustering, synthesis, evaluation, quality gates."""

from typing import Literal

from pydantic import BaseModel, Field

# ── Rule provenance ──────────────────────────────────────────────────────────


class RuleProvenanceRecord(BaseModel):
    """Provenance metadata for a single selected rule."""

    rule_id: str
    topic: str

    # Associative attribution (always present)
    source_case_ids: list[str] = Field(default_factory=list)
    cluster_id: str | None = None
    support_count: int = 0
    support_ratio: float = 0.0
    examples_fixed: list[str] = Field(default_factory=list)
    examples_broken: list[str] = Field(default_factory=list)
    attribution_method: Literal["associative", "causal"] = "associative"

    # Causal fields (populated when ablation results are available)
    ablation_classification: Literal["helpful", "harmful", "neutral", "inconclusive"] | None = None
    ablation_metric_delta: float | None = None
    ablation_changed_cases: int | None = None

    # Flags
    zero_validation_impact: bool = False
    regression_flag: bool = False

    # Optional notes from conflict review / evaluator
    notes: list[str] = Field(default_factory=list)


class RuleProvenanceArtifact(BaseModel):
    """Full provenance artifact for a distillation job."""

    schema_version: Literal["rulekiln.rule_provenance.v1"] = "rulekiln.rule_provenance.v1"
    job_id: str
    strategy_id: str
    rules: list[RuleProvenanceRecord] = Field(default_factory=list)


# ── Rule ablation ────────────────────────────────────────────────────────────


class RuleAblationRecord(BaseModel):
    """Leave-one-rule-out ablation result for a single rule."""

    rule_id: str
    topic: str
    classification: Literal["helpful", "harmful", "neutral", "inconclusive"]
    metric_delta_without_rule: float | None = None
    changed_cases: int | None = None
    primary_metric: str | None = None
    error: str | None = None


class RuleAblationArtifact(BaseModel):
    """Full ablation artifact for a distillation job."""

    schema_version: Literal["rulekiln.rule_ablation.v1"] = "rulekiln.rule_ablation.v1"
    job_id: str
    strategy_id: str
    primary_metric: str | None = None
    records: list[RuleAblationRecord] = Field(default_factory=list)


# ── Pruning-mode comparison ──────────────────────────────────────────────────


class PruningModeRow(BaseModel):
    """One row in a pruning-mode comparison table."""

    mode: Literal["support_count", "utility", "utility_per_token"]
    strategy_id: str
    rule_count: int
    prompt_tokens: int
    primary_metric: str | None = None
    score: float | None = None
    delta_vs_support_count: float | None = None
    evaluated: bool = False


class PruningModeComparison(BaseModel):
    """Cross-mode pruning comparison attached to strategy comparison artifacts."""

    schema_version: Literal["rulekiln.pruning_mode_comparison.v1"] = (
        "rulekiln.pruning_mode_comparison.v1"
    )
    selected_mode: Literal["support_count", "utility", "utility_per_token"]
    rows: list[PruningModeRow] = Field(default_factory=list)

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


class MetricConfidenceInterval(BaseModel):
    """Deterministic confidence interval descriptor for one metric."""

    low: float
    high: float
    method: Literal["bootstrap"] = "bootstrap"
    iterations: int
    seed: int


class PerLabelMetricsRow(BaseModel):
    """Per-label classification metrics row for CSV/JSON artifacts."""

    label: str
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


class TopConfusionRow(BaseModel):
    """Top non-diagonal confusion entry with deterministic example IDs."""

    actual_label: str
    predicted_label: str
    count: int
    example_case_ids: list[str] = Field(default_factory=list)


class RegressedLabelRow(BaseModel):
    """Label-level regression diagnostics between baseline and candidate strategies."""

    label: str
    support: int
    baseline_recall: float
    candidate_recall: float
    recall_delta: float
    baseline_f1: float
    candidate_f1: float
    f1_delta: float
    new_false_negatives: int
    top_predicted_wrong_labels: list[str] = Field(default_factory=list)
    example_case_ids: list[str] = Field(default_factory=list)


class PairedComparisonExample(BaseModel):
    """One baseline-vs-candidate paired comparison example row."""

    schema_version: Literal["rulekiln.paired_comparison_example.v1"] = (
        "rulekiln.paired_comparison_example.v1"
    )
    case_id: str
    change_class: Literal["fixed", "broken", "unchanged"]
    unchanged_status: Literal["both_correct", "both_wrong"] | None = None
    baseline_strategy_id: str
    candidate_strategy_id: str
    expected_label: str
    baseline_prediction: str
    candidate_prediction: str
    baseline_correct: bool
    candidate_correct: bool
    input_text: str
    notes: list[str] = Field(default_factory=list)


class PairedComparisonSummary(BaseModel):
    """Aggregate baseline-vs-candidate paired comparison summary."""

    schema_version: Literal["rulekiln.paired_comparison_summary.v1"] = (
        "rulekiln.paired_comparison_summary.v1"
    )
    baseline_strategy_id: str
    candidate_strategy_id: str
    fixed_count: int
    broken_count: int
    unchanged_correct_count: int
    unchanged_wrong_count: int
    total_cases: int
    net_fix_rate: float | None = None
    net_fix_rate_status: Literal["ok", "no_changed_outcomes"]
    overall_net_fix_rate: float


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
    accuracy_ci_95: MetricConfidenceInterval | None = None
    macro_f1: float | None = None
    macro_f1_ci_95: MetricConfidenceInterval | None = None
    weighted_case_score: float | None = None
    malformed_output_rate: float = 0.0
    per_outcome_precision: dict[str, float] = Field(default_factory=dict)
    per_outcome_recall: dict[str, float] = Field(default_factory=dict)
    per_label_metrics: list[PerLabelMetricsRow] = Field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    top_confusions: list[TopConfusionRow] = Field(default_factory=list)
    regressed_labels: list[RegressedLabelRow] = Field(default_factory=list)
    case_results: list[CaseEvalResult] = Field(default_factory=list)
    violated_rule_counts: dict[str, int] = Field(default_factory=dict)
    failed_assertion_path_counts: dict[str, int] = Field(default_factory=dict)
    failures: list["CaseEvaluationFailure"] = Field(default_factory=list)
    prompt_token_count: int | None = None
    retrieval_failure_count: int = 0
    model_failure_count: int = 0


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


# ── Refinement iteration artifact ────────────────────────────────────────────


class RefinementIterationArtifact(BaseModel):
    """Per-iteration artifact emitted by the closed-loop conflict resolution controller."""

    schema_version: Literal["rulekiln.refinement_iteration.v1"] = (
        "rulekiln.refinement_iteration.v1"
    )
    job_id: str
    iteration: int
    strategy_id: str
    prior_metric: float
    new_metric: float
    improvement: float
    revised_rule_ids: list[str] = Field(default_factory=list)
    stop_reason: str | None = None


# ── Refinement ablation artifact (loop ON vs OFF) ─────────────────────────────


class RefinementAblationRow(BaseModel):
    """One arm of the refinement loop ablation (loop_on or loop_off)."""

    arm: Literal["loop_on", "loop_off"]
    strategy_id: str | None = None
    macro_f1: float | None = None
    macro_f1_ci_low: float | None = None
    macro_f1_ci_high: float | None = None
    regression_rate: float | None = None
    prompt_token_count: int | None = None
    teacher_cost_usd: float | None = None
    iterations_run: int = 0


class RefinementAblationArtifact(BaseModel):
    """Comparison artifact: loop ON vs loop OFF over the same seed and split."""

    schema_version: Literal["rulekiln.refinement_ablation.v1"] = (
        "rulekiln.refinement_ablation.v1"
    )
    benchmark_name: str
    dataset: str
    seed: int
    rows: list[RefinementAblationRow] = Field(default_factory=list)
    loop_helped: bool | None = None
    delta_macro_f1: float | None = None


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
    strategy_evals: dict[str, EvalResult] = Field(default_factory=dict)
    strategy_gates: dict[str, QualityGateResult] = Field(default_factory=dict)
    strategy_prompt_tokens: dict[str, int] = Field(default_factory=dict)
    strategy_metadata: dict[str, dict[str, str | int | float | bool]] = Field(default_factory=dict)
    selected_strategy_id: str | None = None
    selected_strategy_family: str | None = None
    best_distilled_strategy_id: str | None = None
    best_baseline_strategy_id: str | None = None
    best_by_family: dict[str, str] = Field(default_factory=dict)
    paired_comparison: PairedComparisonSummary | None = None
    selected_strategy: str | None = None
    selection_reason: str | None = None
    evaluation_split_warning: str | None = None
    pruning_mode_comparison: PruningModeComparison | None = None
