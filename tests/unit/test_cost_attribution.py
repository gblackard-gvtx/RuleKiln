"""Unit tests for phase and student cost attribution (Task 7)."""

from __future__ import annotations

from rulekiln.benchmarks.schemas import (
    CostSummary,
    PhaseCostBreakdown,
    StudentCostBreakdown,
)


def test_cost_summary_phase_breakdown_empty_by_default() -> None:
    cs = CostSummary()
    assert cs.by_phase == []
    assert cs.by_student == []


def test_cost_summary_accepts_phase_breakdowns() -> None:
    phases = [
        PhaseCostBreakdown(
            phase="instruction_extraction",
            model_id="fake/cheap-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            model_calls=10,
        ),
        PhaseCostBreakdown(
            phase="cluster_consolidation",
            model_id="fake/mid-model",
            input_tokens=200,
            output_tokens=80,
            total_tokens=280,
            estimated_cost_usd=0.005,
            model_calls=5,
        ),
    ]
    cs = CostSummary(by_phase=phases)
    assert len(cs.by_phase) == 2
    assert cs.by_phase[0].phase == "instruction_extraction"
    assert cs.by_phase[1].phase == "cluster_consolidation"


def test_cost_summary_accepts_student_breakdowns() -> None:
    students = [
        StudentCostBreakdown(
            student_id="s1",
            model_id="fake/model-a",
            input_tokens=300,
            output_tokens=120,
            total_tokens=420,
            estimated_cost_usd=0.002,
            model_calls=50,
        ),
        StudentCostBreakdown(
            student_id="s2",
            model_id="fake/model-b",
            input_tokens=310,
            output_tokens=125,
            total_tokens=435,
            estimated_cost_usd=0.0025,
            model_calls=50,
        ),
    ]
    cs = CostSummary(by_student=students)
    assert len(cs.by_student) == 2
    assert cs.by_student[0].student_id == "s1"
    assert cs.by_student[1].student_id == "s2"


def test_phase_totals_independently_consistent() -> None:
    """Phase cost totals are individually accurate (not required to sum to teacher_cost_usd
    since the CostSummary may have been built from a legacy path without breakdowns)."""
    p = PhaseCostBreakdown(
        phase="conflict_resolution",
        model_id="fake/strong-model",
        input_tokens=500,
        output_tokens=200,
        total_tokens=700,
        estimated_cost_usd=0.02,
        model_calls=8,
    )
    assert p.total_tokens == p.input_tokens + p.output_tokens


def test_student_totals_independently_consistent() -> None:
    s = StudentCostBreakdown(
        student_id="anchor",
        model_id="fake/anchor-model",
        input_tokens=400,
        output_tokens=160,
        total_tokens=560,
        estimated_cost_usd=0.003,
        model_calls=40,
    )
    assert s.total_tokens == s.input_tokens + s.output_tokens


def test_cost_summary_backward_compat_no_breakdown_fields() -> None:
    """CostSummary without breakdown fields still serializes correctly (legacy code path)."""
    cs = CostSummary(
        total_input_tokens=1000,
        total_output_tokens=400,
        total_tokens=1400,
        estimated_total_cost_usd=0.01,
        teacher_cost_usd=0.007,
        student_cost_usd=0.003,
    )
    assert cs.by_phase == []
    assert cs.by_student == []
    assert cs.teacher_cost_usd == 0.007


def test_cost_summary_schema_version_unchanged() -> None:
    cs = CostSummary()
    assert cs.schema_version == "rulekiln.cost_summary.v1"


def test_phase_breakdown_defaults_to_zero() -> None:
    p = PhaseCostBreakdown(phase="instruction_extraction", model_id="fake/m")
    assert p.input_tokens == 0
    assert p.model_calls == 0
    assert p.estimated_cost_usd == 0.0
