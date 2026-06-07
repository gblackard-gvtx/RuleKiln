"""PipelineArtifactsState: mutable inter-stage result state for one pipeline run.

Accumulates eval results, compiled prompts, quality-gate outcomes, and other
derived data as the pipeline progresses through the evaluation phase.
Passed by reference through each evaluation stage so every service can read
prior results and write its own.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rulekiln.pipeline.quality_gates import QualityGateResult
from rulekiln.pipeline.statistics import PairedComparisonArtifacts
from rulekiln.schemas.pipeline import EvalResult


class PipelineArtifactsState(BaseModel):
    """Mutable per-run state that accumulates across the evaluation pipeline.

    Created once in _run() after compile_prompts completes and threaded
    through every evaluation and export stage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Baseline prompts ──────────────────────────────────────────────────
    baseline_prompt_text: str = ""
    baseline_prompt_source: str = "compiled"
    baseline_prompt_tokens: int = 0

    # ── Baseline eval results ─────────────────────────────────────────────
    baseline_eval: EvalResult | None = None
    baseline_scaffold_eval: EvalResult | None = None
    baseline_variant_evals: dict[str, EvalResult] = Field(default_factory=dict)
    baseline_variant_prompts: dict[str, str] = Field(default_factory=dict)

    # ── Per-strategy eval results (all strategies, keyed by strategy name) ─
    eval_map: dict[str, EvalResult] = Field(default_factory=dict)

    # ── Per-strategy prompt metadata ──────────────────────────────────────
    strategy_prompt_tokens: dict[str, int] = Field(default_factory=dict)
    strategy_metadata: dict[str, dict[str, str | int | float | bool]] = Field(
        default_factory=dict
    )

    # ── Compiled prompts (baseline variants + distilled, keyed by strategy) ─
    compiled_prompts: dict[str, str] = Field(default_factory=dict)
    token_counts: dict[str, int] = Field(default_factory=dict)

    # ── Quality gate results ──────────────────────────────────────────────
    gate_map: dict[str, QualityGateResult] = Field(default_factory=dict)
    paired_comparison_artifacts: PairedComparisonArtifacts | None = None
    classroom_student_evals: dict[str, dict[str, EvalResult]] = Field(default_factory=dict)
