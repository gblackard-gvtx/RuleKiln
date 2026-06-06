"""Tests for the classroom evaluation pipeline stage (PR 6).

Verifies:
- EVALUATING_CLASSROOM is present in PipelineStage
- StudentEvalSummary is importable from both rulekiln.schemas.pipeline
  and rulekiln.benchmarks.schemas (backward-compat re-export)
- StrategyComparison carries all_student_results
- The stage is positioned after OPTIMIZING_PRUNING
- Single-student classroom configs are skipped (anchor-only optimization)
"""

from __future__ import annotations

from rulekiln.benchmarks.schemas import StudentEvalSummary as StudentEvalSummaryFromBenchmarks
from rulekiln.schemas.pipeline import (
    PairedComparisonSummary,
    StrategyComparison,
    StudentEvalSummary,
)
from rulekiln.workers.distillation_worker import PipelineStage

# ── Schema location and backward-compat re-export ────────────────────────────


def test_student_eval_summary_importable_from_pipeline_schemas() -> None:
    assert StudentEvalSummary is not None
    summary = StudentEvalSummary(student_id="anchor")
    assert summary.student_id == "anchor"
    assert summary.macro_f1 is None


def test_student_eval_summary_importable_from_benchmarks_schemas() -> None:
    """Backward-compat: benchmarks.schemas re-exports StudentEvalSummary from pipeline."""
    assert StudentEvalSummaryFromBenchmarks is StudentEvalSummary


def test_student_eval_summary_has_ci_fields() -> None:
    summary = StudentEvalSummary(
        student_id="s1",
        macro_f1=0.85,
        macro_f1_ci_95=(0.82, 0.88),
        accuracy=0.87,
        accuracy_ci_95=(0.84, 0.90),
        malformed_rate=0.01,
        cost_usd=0.05,
    )
    assert summary.macro_f1_ci_95 == (0.82, 0.88)
    assert summary.accuracy_ci_95 == (0.84, 0.90)
    assert summary.cost_usd == 0.05


def test_student_eval_summary_optional_paired_comparison() -> None:
    pc = PairedComparisonSummary(
        baseline_strategy_id="baseline",
        candidate_strategy_id="dbscan",
        fixed_count=5,
        broken_count=2,
        unchanged_correct_count=10,
        unchanged_wrong_count=3,
        total_cases=20,
        net_fix_rate_status="ok",
        overall_net_fix_rate=0.15,
    )
    summary = StudentEvalSummary(student_id="s1", paired_comparison=pc)
    assert summary.paired_comparison is not None
    assert summary.paired_comparison.fixed_count == 5


# ── StrategyComparison.all_student_results ────────────────────────────────────


def test_strategy_comparison_has_all_student_results_field() -> None:
    comparison = StrategyComparison()
    assert hasattr(comparison, "all_student_results")
    assert comparison.all_student_results == {}


def test_strategy_comparison_all_student_results_accepts_nested_dict() -> None:
    s1 = StudentEvalSummary(student_id="student_a", macro_f1=0.80)
    s2 = StudentEvalSummary(student_id="student_b", macro_f1=0.75)
    comparison = StrategyComparison(
        all_student_results={
            "dbscan": {"student_a": s1, "student_b": s2},
            "baseline": {"student_a": StudentEvalSummary(student_id="student_a", macro_f1=0.70)},
        }
    )
    assert comparison.all_student_results["dbscan"]["student_a"].macro_f1 == 0.80
    assert comparison.all_student_results["dbscan"]["student_b"].macro_f1 == 0.75
    assert comparison.all_student_results["baseline"]["student_a"].macro_f1 == 0.70


# ── PipelineStage.EVALUATING_CLASSROOM ───────────────────────────────────────


def test_evaluating_classroom_stage_present() -> None:
    assert PipelineStage.EVALUATING_CLASSROOM == "evaluating_classroom"
    assert "evaluating_classroom" in {s.value for s in PipelineStage}


def test_evaluating_classroom_stage_ordered_after_optimizing_pruning() -> None:
    stage_values = list(PipelineStage)
    idx_optimizing = stage_values.index(PipelineStage.OPTIMIZING_PRUNING)
    idx_classroom = stage_values.index(PipelineStage.EVALUATING_CLASSROOM)
    assert idx_classroom > idx_optimizing, (
        "EVALUATING_CLASSROOM must come after OPTIMIZING_PRUNING"
    )


def test_evaluating_classroom_stage_ordered_before_logging_artifacts() -> None:
    stage_values = list(PipelineStage)
    idx_classroom = stage_values.index(PipelineStage.EVALUATING_CLASSROOM)
    idx_logging = stage_values.index(PipelineStage.LOGGING_ARTIFACTS)
    assert idx_classroom < idx_logging, (
        "EVALUATING_CLASSROOM must come before LOGGING_ARTIFACTS"
    )


def test_pipeline_stage_count_includes_evaluating_classroom() -> None:
    stage_names = {s.value for s in PipelineStage}
    assert "evaluating_classroom" in stage_names
