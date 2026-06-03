"""Unit tests for pruning-mode comparison rendering in benchmark reports."""

from __future__ import annotations

from datetime import UTC, datetime

from rulekiln.benchmarks.reporting import render_summary_markdown
from rulekiln.benchmarks.schemas import (
    BenchmarkManifest,
    BenchmarkStrategyComparison,
    DatasetManifest,
)
from rulekiln.schemas.pipeline import (
    EvalResult,
    PruningModeComparison,
    PruningModeRow,
)


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_name="banking77",
        run_id="test-run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        git_commit="abc123",
        rulekiln_version="0.1.0",
        python_version="3.12",
        dataset_name="banking77",
        seed=42,
        teacher_model="gpt-4",
        student_model="gpt-3.5",
        embedding_model="text-embedding-ada",
    )


def _dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_name="banking77",
        source="fixture",
        profile="smoke",
        seed=42,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _eval(accuracy: float = 0.8, macro_f1: float = 0.75) -> EvalResult:
    return EvalResult(
        strategy="dbscan",
        model="gpt-3.5",
        split="test",
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def _comparison(
    pruning_mode_comparison: PruningModeComparison | None = None,
) -> BenchmarkStrategyComparison:
    return BenchmarkStrategyComparison(
        primary_metric="macro_f1",
        baseline_eval=_eval(accuracy=0.7, macro_f1=0.65),
        rulekiln_eval=_eval(),
        baseline_score=0.65,
        rulekiln_score=0.75,
        delta_vs_baseline=0.10,
        selected_strategy="dbscan",
        selection_reason="best_distilled",
        pruning_mode_comparison=pruning_mode_comparison,
    )


def test_render_without_pruning_mode_comparison() -> None:
    md = render_summary_markdown(
        _manifest(),
        _dataset_manifest(),
        _comparison(pruning_mode_comparison=None),
        reproduction_command="uv run rulekiln-bench banking77",
    )
    assert "## Pruning Mode Comparison" not in md
    assert "## Regressed Labels" in md


def test_render_with_pruning_mode_comparison_single_row() -> None:
    comparison = PruningModeComparison(
        selected_mode="support_count",
        rows=[
            PruningModeRow(
                mode="support_count",
                strategy_id="dbscan",
                rule_count=15,
                prompt_tokens=1200,
                primary_metric="macro_f1",
                score=0.75,
                delta_vs_support_count=0.0,
                evaluated=True,
            )
        ],
    )
    md = render_summary_markdown(
        _manifest(),
        _dataset_manifest(),
        _comparison(pruning_mode_comparison=comparison),
        reproduction_command="uv run rulekiln-bench banking77",
    )
    assert "## Pruning Mode Comparison" in md
    assert "selected_mode: support_count" in md
    assert "support_count" in md
    assert "0.750000" in md


def test_render_with_multi_row_pruning_mode_comparison() -> None:
    comparison = PruningModeComparison(
        selected_mode="utility",
        rows=[
            PruningModeRow(
                mode="support_count",
                strategy_id="dbscan",
                rule_count=15,
                prompt_tokens=1200,
                score=0.75,
                delta_vs_support_count=0.0,
                evaluated=True,
            ),
            PruningModeRow(
                mode="utility",
                strategy_id="dbscan_pruning_utility",
                rule_count=12,
                prompt_tokens=980,
                score=0.77,
                delta_vs_support_count=0.02,
                evaluated=True,
            ),
            PruningModeRow(
                mode="utility_per_token",
                strategy_id="dbscan_pruning_utility_per_token",
                rule_count=10,
                prompt_tokens=800,
                score=0.76,
                delta_vs_support_count=0.01,
                evaluated=True,
            ),
        ],
    )
    md = render_summary_markdown(
        _manifest(),
        _dataset_manifest(),
        _comparison(pruning_mode_comparison=comparison),
        reproduction_command="uv run rulekiln-bench banking77",
    )
    assert "## Pruning Mode Comparison" in md
    assert "selected_mode: utility" in md
    assert "utility_per_token" in md
    assert "+0.020000" in md
    assert "+0.010000" in md


def test_render_pruning_mode_table_headers_present() -> None:
    comparison = PruningModeComparison(
        selected_mode="support_count",
        rows=[
            PruningModeRow(
                mode="support_count",
                strategy_id="dbscan",
                rule_count=5,
                prompt_tokens=500,
                evaluated=True,
            )
        ],
    )
    md = render_summary_markdown(
        _manifest(),
        _dataset_manifest(),
        _comparison(pruning_mode_comparison=comparison),
        reproduction_command="uv run rulekiln-bench",
    )
    assert "| mode |" in md
    assert "| rules |" in md
    assert "| prompt_tokens |" in md
    assert "| delta_vs_support_count |" in md
