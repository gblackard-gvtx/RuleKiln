"""Unit tests for classroom-aware benchmark phases 1–4.

Covers:
- Phase 1.3: cross-strategy × student matrix rendering
- Phase 2.2: non-LLM baseline section rendering
- Phase 3.1: CI fields on StudentEvalSummary from EvalResult
- Phase 3.2: paired_comparison on StudentEvalSummary
- Phase 3.3: write_per_student_artifacts
- Phase 3.4: lift table rendering
- Phase 4.3: new classroom pruning modes
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rulekiln.benchmarks.reporting import (
    _render_lift_table_section,
    _render_non_llm_baselines_section,
    _render_strategy_student_matrix_section,
    render_summary_markdown,
    student_eval_summary_from_eval_result,
    write_per_student_artifacts,
)
from rulekiln.benchmarks.schemas import (
    BenchmarkManifest,
    BenchmarkStrategyComparison,
    DatasetManifest,
    StudentEvalSummary,
)
from rulekiln.pipeline.rule_pruning import (
    ClassroomUtilitySignals,
    UtilitySignals,
    prune_rules,
)
from rulekiln.schemas.pipeline import (
    EvalResult,
    MetricConfidenceInterval,
    PairedComparisonSummary,
    RuleProvenanceReport,
    RuleStudentUtility,
    SynthesizedRuleSchema,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_name="test",
        run_id="run-1",
        git_commit="abc",
        rulekiln_version="0.1.0",
        python_version="3.13",
        dataset_name="test_ds",
        seed=42,
        teacher_model="fake",
        student_model="fake",
        embedding_model="fake",
    )


def _eval_result(strategy: str = "dbscan", macro_f1: float = 0.75) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="fake",
        split="test",
        macro_f1=macro_f1,
        accuracy=macro_f1 + 0.05,
        weighted_case_score=macro_f1,
    )


def _comparison(**kwargs: object) -> BenchmarkStrategyComparison:
    return BenchmarkStrategyComparison(
        primary_metric="macro_f1",
        baseline_eval=_eval_result("baseline", 0.65),
        rulekiln_eval=_eval_result("dbscan"),
        baseline_score=0.65,
        rulekiln_score=0.75,
        delta_vs_baseline=0.10,
        selected_strategy="dbscan",
        selection_reason="higher macro_f1",
        **kwargs,  # type: ignore[arg-type]
    )


def _dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_name="test_ds",
        source="fixture",
        profile="smoke",
        seed=42,
    )


def _student_summary(sid: str, f1: float) -> StudentEvalSummary:
    return StudentEvalSummary(student_id=sid, macro_f1=f1, accuracy=f1 + 0.05)


def _make_rule(rule_id: str, support: int = 5) -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=rule_id,
        support_count=support,
        support_ratio=support / 10.0,
        estimated_token_count=50,
    )


# ── Phase 1.3: cross-strategy × student matrix ──────────────────────────────


def test_strategy_matrix_empty_when_no_results() -> None:
    lines = _render_strategy_student_matrix_section({})
    assert lines == []


def test_strategy_matrix_single_strategy_single_student() -> None:
    results = {"baseline": {"s1": _student_summary("s1", 0.70)}}
    lines = _render_strategy_student_matrix_section(results)
    joined = "\n".join(lines)
    assert "Strategy × Student Results (macro_f1)" in joined
    assert "baseline" in joined
    assert "s1" in joined
    assert "0.7000" in joined


def test_strategy_matrix_multi_strategy_multi_student() -> None:
    results = {
        "baseline_scaffold": {
            "qwen_7b": _student_summary("qwen_7b", 0.41),
            "haiku": _student_summary("haiku", 0.67),
        },
        "rulekiln_dbscan": {
            "qwen_7b": _student_summary("qwen_7b", 0.71),
            "haiku": _student_summary("haiku", 0.88),
        },
    }
    lines = _render_strategy_student_matrix_section(results)
    joined = "\n".join(lines)
    assert "baseline_scaffold" in joined
    assert "rulekiln_dbscan" in joined
    assert "qwen_7b" in joined
    assert "haiku" in joined
    assert "0.4100" in joined
    assert "0.8800" in joined


def test_strategy_matrix_display_names_used_as_column_headers() -> None:
    results = {"baseline": {"s1": _student_summary("s1", 0.70)}}
    lines = _render_strategy_student_matrix_section(
        results,
        student_display_names={"s1": "Qwen 7B"},
    )
    joined = "\n".join(lines)
    assert "Qwen 7B" in joined
    assert "| s1 |" not in joined


def test_strategy_matrix_student_order_followed() -> None:
    results = {
        "baseline": {
            "zebra": _student_summary("zebra", 0.60),
            "alpha": _student_summary("alpha", 0.50),
        }
    }
    lines = _render_strategy_student_matrix_section(results, student_order=["zebra", "alpha"])
    # The table header row starts with "| Strategy |"
    header = next(line for line in lines if line.startswith("| Strategy |"))
    # 'zebra' should appear before 'alpha' in the header (overriding alphabetical order)
    assert header.index("zebra") < header.index("alpha")


def test_strategy_matrix_missing_student_shown_as_dash() -> None:
    results = {
        "baseline": {"s1": _student_summary("s1", 0.70)},
        "rulekiln": {},  # s1 has no result here
    }
    lines = _render_strategy_student_matrix_section(results)
    rulekiln_row = [line for line in lines if "rulekiln" in line][0]
    assert "—" in rulekiln_row


def test_strategy_matrix_renders_in_summary_markdown() -> None:
    s1 = _student_summary("s1", 0.71)
    s2 = _student_summary("s2", 0.88)
    c = _comparison(
        all_student_results={
            "baseline_scaffold": {
                "s1": _student_summary("s1", 0.41),
                "s2": _student_summary("s2", 0.67),
            },
            "rulekiln_dbscan": {"s1": s1, "s2": s2},
        },
        best_baseline_strategy_id="baseline_scaffold",
    )
    md = render_summary_markdown(
        _manifest(), _dataset_manifest(), c, reproduction_command="uv run rulekiln-benchmark"
    )
    assert "Strategy × Student Results" in md
    assert "baseline_scaffold" in md
    assert "rulekiln_dbscan" in md


def test_summary_markdown_falls_back_to_single_student_matrix() -> None:
    """When all_student_results is empty, fall back to student_results."""
    s1 = _student_summary("anchor", 0.75)
    c = _comparison(student_results={"anchor": s1})
    md = render_summary_markdown(
        _manifest(), _dataset_manifest(), c, reproduction_command="uv run rulekiln-benchmark"
    )
    assert "Student Evaluation Matrix" in md
    assert "Strategy × Student Results" not in md


# ── Phase 2.2: non-LLM baseline section ─────────────────────────────────────


def test_non_llm_baselines_section_empty_when_no_results() -> None:
    lines = _render_non_llm_baselines_section({})
    assert lines == []


def test_non_llm_baselines_section_renders_table() -> None:
    results = {
        "embedding_knn_k5": {"macro_f1": 0.38, "accuracy": 0.40},
        "embedding_centroid": {"macro_f1": 0.33, "accuracy": 0.35},
    }
    lines = _render_non_llm_baselines_section(results)
    joined = "\n".join(lines)
    assert "Non-LLM Baselines (embedding-only)" in joined
    assert "embedding_knn_k5" in joined
    assert "embedding_centroid" in joined
    assert "0.3800" in joined
    assert "0.3300" in joined


def test_non_llm_baselines_handles_none_values() -> None:
    results = {"embedding_centroid": {"macro_f1": None, "accuracy": 0.35}}
    lines = _render_non_llm_baselines_section(results)
    joined = "\n".join(lines)
    assert "—" in joined
    assert "0.3500" in joined


def test_non_llm_baselines_renders_in_summary_markdown() -> None:
    c = _comparison(
        non_llm_baseline_results={
            "embedding_knn_k5": {"macro_f1": 0.38, "accuracy": 0.40},
        }
    )
    md = render_summary_markdown(
        _manifest(), _dataset_manifest(), c, reproduction_command="uv run rulekiln-benchmark"
    )
    assert "Non-LLM Baselines (embedding-only)" in md
    assert "embedding_knn_k5" in md


# ── Phase 3.1: CI fields on StudentEvalSummary from EvalResult ───────────────


def test_student_eval_summary_from_eval_result_basic() -> None:
    er = EvalResult(
        strategy="dbscan",
        model="fake",
        split="test",
        macro_f1=0.75,
        accuracy=0.80,
        malformed_output_rate=0.02,
    )
    summary = student_eval_summary_from_eval_result(er, "s1")
    assert summary.student_id == "s1"
    assert summary.macro_f1 == 0.75
    assert summary.accuracy == 0.80
    assert summary.malformed_rate == 0.02
    assert summary.macro_f1_ci_95 is None
    assert summary.accuracy_ci_95 is None


def test_student_eval_summary_from_eval_result_with_ci() -> None:
    ci = MetricConfidenceInterval(low=0.70, high=0.80, iterations=1000, seed=42)
    er = EvalResult(
        strategy="dbscan",
        model="fake",
        split="test",
        macro_f1=0.75,
        accuracy=0.80,
        macro_f1_ci_95=ci,
        accuracy_ci_95=MetricConfidenceInterval(low=0.76, high=0.84, iterations=1000, seed=42),
    )
    summary = student_eval_summary_from_eval_result(er, "s1", cost_usd=0.05, latency_p95_ms=250.0)
    assert summary.macro_f1_ci_95 == (0.70, 0.80)
    assert summary.accuracy_ci_95 == (0.76, 0.84)
    assert summary.cost_usd == 0.05
    assert summary.latency_p95_ms == 250.0


# ── Phase 3.2: paired_comparison on StudentEvalSummary ───────────────────────


def test_student_eval_summary_stores_paired_comparison() -> None:
    pc = PairedComparisonSummary(
        baseline_strategy_id="baseline_scaffold",
        candidate_strategy_id="rulekiln_dbscan",
        fixed_count=30,
        broken_count=10,
        unchanged_correct_count=200,
        unchanged_wrong_count=60,
        total_cases=300,
        net_fix_rate=0.5,
        net_fix_rate_status="ok",
        overall_net_fix_rate=0.0667,
    )
    s = StudentEvalSummary(student_id="s1", macro_f1=0.75, paired_comparison=pc)
    assert s.paired_comparison is not None
    assert s.paired_comparison.fixed_count == 30
    assert s.paired_comparison.broken_count == 10


# ── Phase 3.3: write_per_student_artifacts ───────────────────────────────────


def test_write_per_student_artifacts_creates_files(tmp_path: Path) -> None:
    from rulekiln.pipeline.statistics import compute_classification_statistics

    actual = ["cat", "dog", "cat", "dog", "cat"]
    predicted = ["cat", "cat", "cat", "dog", "dog"]
    case_ids = ["c1", "c2", "c3", "c4", "c5"]

    stats = compute_classification_statistics(
        actual_labels=actual,
        predicted_labels=predicted,
        case_ids=case_ids,
        bootstrap_enabled=False,
        bootstrap_iterations=100,
        bootstrap_seed=42,
    )
    from rulekiln.schemas.pipeline import CaseEvalResult

    er = EvalResult(
        strategy="rulekiln_dbscan_s1",
        model="fake",
        split="test",
        macro_f1=stats.macro_f1,
        accuracy=stats.accuracy,
        per_label_metrics=stats.per_label_metrics,
        confusion_matrix=stats.confusion_matrix,
        case_results=[CaseEvalResult(case_id=cid, score=1.0, passed=True) for cid in case_ids],
    )

    per_student_root = tmp_path / "per_student"
    written = write_per_student_artifacts(per_student_root, "rulekiln_dbscan", {"s1": er})

    student_dir = per_student_root / "s1"
    assert student_dir.exists()
    eval_file = student_dir / "rulekiln_dbscan_eval.json"
    assert eval_file in written
    assert eval_file.exists()
    assert json.loads(eval_file.read_text())["strategy"] == "rulekiln_dbscan_s1"

    csv_file = student_dir / "rulekiln_dbscan_per_label_metrics.csv"
    assert csv_file in written
    assert csv_file.exists()

    cm_file = student_dir / "rulekiln_dbscan_confusion_matrix.csv"
    assert cm_file in written
    assert cm_file.exists()


# ── Phase 3.4: lift table rendering ──────────────────────────────────────────


def test_lift_table_empty_when_no_results() -> None:
    lines = _render_lift_table_section({}, "baseline_scaffold")
    assert lines == []


def test_lift_table_empty_when_baseline_missing() -> None:
    results = {"rulekiln_dbscan": {"s1": _student_summary("s1", 0.71)}}
    lines = _render_lift_table_section(results, "baseline_scaffold")
    assert lines == []


def test_lift_table_computes_per_student_deltas() -> None:
    results = {
        "baseline_scaffold": {
            "s1": _student_summary("s1", 0.41),
            "s2": _student_summary("s2", 0.67),
        },
        "rulekiln_dbscan": {
            "s1": _student_summary("s1", 0.71),
            "s2": _student_summary("s2", 0.88),
        },
    }
    lines = _render_lift_table_section(results, "baseline_scaffold")
    joined = "\n".join(lines)
    assert "Lift vs. Baseline" in joined
    assert "baseline_scaffold" in joined
    assert "rulekiln_dbscan" in joined
    # +0.30 for s1, +0.21 for s2
    assert "+0.3000" in joined
    assert "+0.2100" in joined


def test_lift_table_emits_classroom_aggregate() -> None:
    results = {
        "baseline_scaffold": {
            "s1": _student_summary("s1", 0.40),
            "s2": _student_summary("s2", 0.60),
        },
        "rulekiln_hdbscan": {
            "s1": _student_summary("s1", 0.70),
            "s2": _student_summary("s2", 0.80),
        },
    }
    lines = _render_lift_table_section(results, "baseline_scaffold")
    joined = "\n".join(lines)
    assert "Classroom aggregate lift" in joined
    assert "mean across students, unweighted" in joined
    # mean(0.30, 0.20) = 0.25
    assert "+0.2500" in joined


def test_lift_table_renders_in_summary_markdown() -> None:
    c = _comparison(
        all_student_results={
            "baseline_scaffold": {"s1": _student_summary("s1", 0.41)},
            "rulekiln_dbscan": {"s1": _student_summary("s1", 0.71)},
        },
        best_baseline_strategy_id="baseline_scaffold",
    )
    md = render_summary_markdown(
        _manifest(), _dataset_manifest(), c, reproduction_command="uv run rulekiln-benchmark"
    )
    assert "Lift vs. Baseline" in md
    assert "Classroom aggregate lift" in md


# ── Phase 4.2: RuleStudentUtility and RuleProvenanceReport schemas ────────────


def test_rule_student_utility_schema() -> None:
    rsu = RuleStudentUtility(
        rule_id="r1",
        student_id="s1",
        fixed_count=10,
        broken_count=2,
        net_utility=8,
        utility_per_token=0.16,
    )
    assert rsu.net_utility == 8
    assert rsu.utility_per_token == pytest.approx(0.16)


def test_rule_provenance_report_schema() -> None:
    rpr = RuleProvenanceReport(
        rule_id="r1",
        topic="topic_1",
        support_count=15,
        source_case_ids=["c1", "c2"],
        student_utility=[
            RuleStudentUtility(
                rule_id="r1",
                student_id="anchor",
                fixed_count=10,
                broken_count=2,
                net_utility=8,
                utility_per_token=0.16,
            ),
        ],
        anchor_net_utility=8,
        mean_classroom_net_utility=6.5,
        worst_student_net_utility=4,
    )
    assert rpr.anchor_net_utility == 8
    assert rpr.mean_classroom_net_utility == pytest.approx(6.5)
    assert rpr.worst_student_net_utility == 4


# ── Phase 4.3: new pruning modes ─────────────────────────────────────────────


def test_anchor_utility_mode_equivalent_to_utility_with_anchor_signals() -> None:
    rules = [
        _make_rule("r1", support=5),
        _make_rule("r2", support=5),
    ]
    signals: UtilitySignals = {"r1": (10, 1), "r2": (3, 5)}

    result_utility = prune_rules(rules, ranking_mode="utility", utility_signals=signals)
    result_anchor = prune_rules(rules, ranking_mode="anchor_utility", utility_signals=signals)

    assert [r.id for r in result_utility.selected] == [r.id for r in result_anchor.selected]


def test_mean_classroom_utility_uses_mean_across_students() -> None:
    rules = [_make_rule("r1", support=5), _make_rule("r2", support=5)]
    # r1: anchor=high, other=low → mean moderate
    # r2: anchor=low, other=high → mean moderate
    # with mean mode, selection should reflect unweighted average
    classroom_signals: ClassroomUtilitySignals = {
        "anchor": {"r1": (10, 0), "r2": (2, 0)},  # r1=10, r2=2
        "student_b": {"r1": (2, 0), "r2": (10, 0)},  # r1=2, r2=10
    }
    result = prune_rules(
        rules,
        ranking_mode="mean_classroom_utility",
        classroom_utility_signals=classroom_signals,
    )
    # mean(r1) = 6, mean(r2) = 6 — equal, order by support_count as tiebreak
    assert len(result.selected) == 2


def test_worst_student_utility_picks_rules_that_help_all_students() -> None:
    rules = [_make_rule("r1", support=5), _make_rule("r2", support=5)]
    # r1: great for anchor but negative for student_b → worst = -3
    # r2: good for both → worst = 5
    classroom_signals: ClassroomUtilitySignals = {
        "anchor": {"r1": (10, 0), "r2": (7, 0)},
        "student_b": {"r1": (0, 3), "r2": (5, 0)},
    }
    result = prune_rules(
        rules, ranking_mode="worst_student_utility", classroom_utility_signals=classroom_signals
    )
    # r2 should rank higher than r1 because worst-student(r1)=-3 < worst-student(r2)=5
    assert result.selected[0].id == "r2"
    assert result.selected[1].id == "r1"


def test_worst_student_utility_beats_anchor_utility_when_negative() -> None:
    """High anchor utility but negative for one student ranks lower under worst_student."""
    rules = [_make_rule("r_high_anchor", support=5), _make_rule("r_balanced", support=5)]
    classroom_signals: ClassroomUtilitySignals = {
        "anchor": {"r_high_anchor": (20, 0), "r_balanced": (5, 0)},
        "other": {"r_high_anchor": (0, 5), "r_balanced": (5, 0)},  # r_high_anchor hurts 'other'
    }
    worst_result = prune_rules(
        rules, ranking_mode="worst_student_utility", classroom_utility_signals=classroom_signals
    )
    anchor_result = prune_rules(
        rules,
        ranking_mode="anchor_utility",
        utility_signals={"r_high_anchor": (20, 0), "r_balanced": (5, 0)},
    )
    # Under worst_student, r_balanced ranks first; under anchor_utility, r_high_anchor ranks first
    assert worst_result.selected[0].id == "r_balanced"
    assert anchor_result.selected[0].id == "r_high_anchor"


def test_single_student_classroom_mean_equals_anchor() -> None:
    """mean_classroom_utility == anchor_utility when there is one student."""
    rules = [_make_rule("r1", support=5), _make_rule("r2", support=3)]
    signals: UtilitySignals = {"r1": (10, 1), "r2": (4, 2)}
    classroom_signals: ClassroomUtilitySignals = {"anchor": signals}

    anchor_result = prune_rules(rules, ranking_mode="anchor_utility", utility_signals=signals)
    mean_result = prune_rules(
        rules, ranking_mode="mean_classroom_utility", classroom_utility_signals=classroom_signals
    )
    assert [r.id for r in anchor_result.selected] == [r.id for r in mean_result.selected]
