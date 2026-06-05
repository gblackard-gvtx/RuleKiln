"""Unit tests for classroom-aware benchmark schema additions (Task 5)."""

from __future__ import annotations

from rulekiln.benchmarks.reporting import _render_student_matrix_section, render_summary_markdown
from rulekiln.benchmarks.schemas import (
    BenchmarkManifest,
    BenchmarkStrategyComparison,
    CostSummary,
    DatasetManifest,
    StudentEvalSummary,
)
from rulekiln.schemas.classroom import ClassroomConfig, PhaseTeacherConfig, TeacherConfig
from rulekiln.schemas.pipeline import EvalResult


def _eval_result(strategy: str = "dbscan") -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="fake",
        split="validation",
        macro_f1=0.75,
        accuracy=0.80,
        weighted_case_score=0.80,
        per_outcome_precision={},
        per_outcome_recall={},
        malformed_output_rate=0.0,
        confusion_matrix={},
        case_results=[],
    )


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


def _comparison() -> BenchmarkStrategyComparison:
    return BenchmarkStrategyComparison(
        primary_metric="macro_f1",
        baseline_eval=_eval_result("baseline"),
        rulekiln_eval=_eval_result("dbscan"),
        baseline_score=0.70,
        rulekiln_score=0.75,
        delta_vs_baseline=0.05,
        selected_strategy="dbscan",
        selection_reason="higher macro_f1",
    )


# ── schema_version fields ────────────────────────────────────────────────────


def test_benchmark_manifest_schema_version() -> None:
    m = _manifest()
    assert m.schema_version == "rulekiln.benchmark_manifest.v3"


def test_strategy_comparison_schema_version() -> None:
    c = _comparison()
    assert c.schema_version == "rulekiln.strategy_comparison.v2"


def test_cost_summary_schema_version() -> None:
    cs = CostSummary()
    assert cs.schema_version == "rulekiln.cost_summary.v1"


def test_student_eval_summary_schema_version() -> None:
    s = StudentEvalSummary(student_id="s1")
    assert s.schema_version == "rulekiln.student_eval_summary.v1"


# ── BenchmarkManifest additions ──────────────────────────────────────────────


def test_manifest_has_classroom_fields_defaulting_to_none() -> None:
    m = _manifest()
    assert m.teacher_config is None
    assert m.classroom_config is None
    assert m.extraction_cache_hits == 0
    assert m.extraction_cache_misses == 0
    assert m.conflict_resolution_anchor_id is None
    # Phase 1.1 provenance fields
    assert m.teacher_provider_profile is None
    assert m.student_provider_profiles == []
    assert m.embedding_provider_profile is None
    assert m.max_concurrent_students is None
    assert m.max_concurrent_cases is None


def test_manifest_v3_provider_fields() -> None:
    m = _manifest()
    m2 = m.model_copy(
        update={
            "teacher_provider_profile": "anthropic-default",
            "student_provider_profiles": ["openai-qwen7b", "openai-qwen14b"],
            "embedding_provider_profile": "openai-embed",
            "max_concurrent_students": 4,
            "max_concurrent_cases": 16,
        }
    )
    assert m2.teacher_provider_profile == "anthropic-default"
    assert m2.student_provider_profiles == ["openai-qwen7b", "openai-qwen14b"]
    assert m2.embedding_provider_profile == "openai-embed"
    assert m2.max_concurrent_students == 4
    assert m2.max_concurrent_cases == 16


def test_manifest_accepts_teacher_config() -> None:
    tc = TeacherConfig(default=PhaseTeacherConfig(provider="fake", model="m"))
    m = _manifest()
    m2 = m.model_copy(update={"teacher_config": tc})
    assert m2.teacher_config is not None
    assert m2.teacher_config.default.provider == "fake"


def test_manifest_accepts_classroom_config() -> None:
    cc = ClassroomConfig.from_provider_model("fake", "m")
    m = _manifest()
    m2 = m.model_copy(update={"classroom_config": cc})
    assert m2.classroom_config is not None


def test_manifest_cache_hit_miss_fields() -> None:
    m = _manifest()
    m2 = m.model_copy(update={"extraction_cache_hits": 10, "extraction_cache_misses": 2})
    assert m2.extraction_cache_hits == 10
    assert m2.extraction_cache_misses == 2


# ── StrategyResult (BenchmarkStrategyComparison) additions ──────────────────


def test_comparison_student_results_default_empty() -> None:
    c = _comparison()
    assert c.student_results == {}


def test_comparison_accepts_student_results() -> None:
    s1 = StudentEvalSummary(student_id="s1", macro_f1=0.75, accuracy=0.80, malformed_rate=0.01)
    s2 = StudentEvalSummary(student_id="s2", macro_f1=0.60, accuracy=0.65, malformed_rate=0.02)
    c = _comparison()
    c2 = c.model_copy(update={"student_results": {"s1": s1, "s2": s2}})
    assert "s1" in c2.student_results
    assert "s2" in c2.student_results
    assert c2.student_results["s1"].macro_f1 == 0.75


# ── summary.md matrix table ──────────────────────────────────────────────────


def test_student_matrix_section_renders_all_students() -> None:
    summaries = {
        "s1": StudentEvalSummary(
            student_id="s1", macro_f1=0.80, accuracy=0.82, malformed_rate=0.01
        ),  # noqa: E501
        "s2": StudentEvalSummary(
            student_id="s2", macro_f1=0.70, accuracy=0.72, malformed_rate=0.02
        ),  # noqa: E501
    }
    lines = _render_student_matrix_section(summaries)
    joined = "\n".join(lines)
    assert "s1" in joined
    assert "s2" in joined
    assert "macro_f1" in joined
    assert "accuracy" in joined


def test_student_matrix_empty_when_no_students() -> None:
    lines = _render_student_matrix_section({})
    assert lines == []


def test_render_summary_markdown_includes_matrix_when_students_present() -> None:
    s1 = StudentEvalSummary(student_id="anchor", macro_f1=0.75, accuracy=0.80)
    c = _comparison()
    c2 = c.model_copy(update={"student_results": {"anchor": s1}})
    m = _manifest()
    ds = DatasetManifest(
        dataset_name="ds",
        source="fixture",
        profile="smoke",
        seed=42,
    )
    md = render_summary_markdown(m, ds, c2, reproduction_command="uv run rulekiln-benchmark")
    assert "Student Evaluation Matrix" in md
    assert "anchor" in md


def test_render_summary_markdown_no_matrix_when_no_students() -> None:
    c = _comparison()
    m = _manifest()
    ds = DatasetManifest(
        dataset_name="ds",
        source="fixture",
        profile="smoke",
        seed=42,
    )
    md = render_summary_markdown(m, ds, c, reproduction_command="uv run rulekiln-benchmark")
    assert "Student Evaluation Matrix" not in md


# ── backward compat: existing single-student instantiation still works ────────


def test_manifest_backward_compat_no_new_fields_required() -> None:
    """Old code that only sets legacy fields still instantiates BenchmarkManifest."""
    m = BenchmarkManifest(
        benchmark_name="b",
        run_id="r",
        git_commit="g",
        rulekiln_version="0.1",
        python_version="3.13",
        dataset_name="ds",
        seed=1,
        teacher_model="t",
        student_model="s",
        embedding_model="e",
    )
    assert m.teacher_config is None
    assert m.classroom_config is None


# ── all_student_results cross-strategy matrix ──────────────────────────────────


def test_comparison_all_student_results_default_empty() -> None:
    c = _comparison()
    assert c.all_student_results == {}


def test_comparison_accepts_all_student_results() -> None:
    s1 = StudentEvalSummary(student_id="s1", macro_f1=0.75)
    s2 = StudentEvalSummary(student_id="s2", macro_f1=0.60)
    c = _comparison()
    c2 = c.model_copy(
        update={
            "all_student_results": {
                "baseline_scaffold": {"s1": s1, "s2": s2},
                "rulekiln_dbscan": {"s1": s1, "s2": s2},
            }
        }
    )
    assert "baseline_scaffold" in c2.all_student_results
    assert "s1" in c2.all_student_results["rulekiln_dbscan"]


def test_comparison_non_llm_baseline_results_default_empty() -> None:
    c = _comparison()
    assert c.non_llm_baseline_results == {}


def test_comparison_accepts_non_llm_baseline_results() -> None:
    c = _comparison()
    c2 = c.model_copy(
        update={
            "non_llm_baseline_results": {
                "embedding_knn_k5": {"macro_f1": 0.38, "accuracy": 0.40},
                "embedding_centroid": {"macro_f1": 0.33, "accuracy": 0.35},
            }
        }
    )
    assert "embedding_knn_k5" in c2.non_llm_baseline_results
    assert c2.non_llm_baseline_results["embedding_centroid"]["macro_f1"] == 0.33


# ── StudentEvalSummary paired_comparison field ────────────────────────────────


def test_student_eval_summary_paired_comparison_defaults_none() -> None:
    s = StudentEvalSummary(student_id="s1")
    assert s.paired_comparison is None
