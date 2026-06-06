"""PipelinePhase type and PhasePlan model.

PipelinePhase is defined here (not in distillation_worker) to keep the pipeline
package self-contained.  distillation_worker re-exports it for backward compat.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

PipelinePhase = Literal[
    "full",
    "validate_project",
    "compile_prompts",
    "extraction_batch_submit",
    "extraction_batch_collect",
    "evaluate_baseline",
    "evaluate_dbscan",
    "evaluate_hdbscan",
    "aggregate_evaluation_report",
]

_COMPILE_PHASES: frozenset[str] = frozenset(
    {"full", "compile_prompts", "extraction_batch_submit", "extraction_batch_collect"}
)
_POST_EXTRACTION_PHASES: frozenset[str] = frozenset(
    {"full", "compile_prompts", "extraction_batch_collect"}
)


class PhasePlan(BaseModel):
    """Which pipeline sub-phases to execute for a given PipelinePhase value."""

    run_validate: bool
    run_compile_phase: bool
    run_post_extraction: bool
    run_baseline_eval: bool
    run_aggregate: bool

    @classmethod
    def from_phase(cls, phase: PipelinePhase) -> PhasePlan:
        return cls(
            run_validate=phase in {"full", "validate_project", "compile_prompts"},
            run_compile_phase=phase in _COMPILE_PHASES,
            run_post_extraction=phase in _POST_EXTRACTION_PHASES,
            run_baseline_eval=phase in {"full", "evaluate_baseline"},
            run_aggregate=phase in {"full", "aggregate_evaluation_report"},
        )
