"""Writers for benchmark artifacts and snapshots."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rulekiln.artifacts.writer import (
    write_confusion_matrix_csv as write_confusion_matrix_csv_shared,
)
from rulekiln.artifacts.writer import (
    write_paired_comparison_artifacts,
)
from rulekiln.artifacts.writer import (
    write_per_label_metrics_csv as write_per_label_metrics_csv_shared,
)
from rulekiln.artifacts.writer import (
    write_top_confusions_markdown as write_top_confusions_markdown_shared,
)
from rulekiln.benchmarks.schemas import (
    BenchmarkManifest,
    BenchmarkStrategyComparison,
    DatasetManifest,
    StudentEvalSummary,
)
from rulekiln.pipeline.statistics import PairedComparisonArtifacts
from rulekiln.schemas.pipeline import (
    EvalResult,
    MetricConfidenceInterval,
    PerLabelMetricsRow,
    PruningModeComparison,
    RegressedLabelRow,
    TopConfusionRow,
)

_SNAPSHOT_START = "<!-- RULEKILN_BENCHMARK_SNAPSHOT_START -->"
_SNAPSHOT_END = "<!-- RULEKILN_BENCHMARK_SNAPSHOT_END -->"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_eval(path: Path, eval_result: EvalResult) -> Path:
    """Write an EvalResult JSON file."""
    return _write_json(path, eval_result.model_dump(mode="json"))


def write_strategy_comparison(path: Path, comparison: BenchmarkStrategyComparison) -> Path:
    """Write strategy comparison JSON file."""
    return _write_json(path, comparison.model_dump(mode="json"))


def write_benchmark_manifest(path: Path, manifest: BenchmarkManifest) -> Path:
    """Write benchmark manifest JSON file."""
    return _write_json(path, manifest.model_dump(mode="json"))


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> Path:
    """Write dataset manifest JSON file."""
    return _write_json(path, manifest.model_dump(mode="json"))


def write_confusion_matrix_csv(path: Path, eval_result: EvalResult) -> Path:
    """Write strict confusion matrix CSV (actual_label,predicted_label,count)."""
    return write_confusion_matrix_csv_shared(path, eval_result.confusion_matrix)


def write_per_label_metrics_csv(path: Path, rows: list[PerLabelMetricsRow]) -> Path:
    """Write strict per-label metrics CSV."""
    return write_per_label_metrics_csv_shared(path, rows)


def write_top_confusions_markdown(path: Path, rows: list[TopConfusionRow]) -> Path:
    """Write top confusions markdown."""
    return write_top_confusions_markdown_shared(path, rows)


def write_paired_comparison(path: Path, paired_comparison: PairedComparisonArtifacts) -> list[Path]:
    """Write paired comparison JSONL and summary files."""
    return write_paired_comparison_artifacts(path, paired_comparison)


def _format_ci(metric_ci: MetricConfidenceInterval | None) -> str:
    if metric_ci is None:
        return "null"
    return json.dumps(metric_ci.model_dump(mode="json"), ensure_ascii=False)


def _render_pruning_mode_comparison_section(
    comparison: PruningModeComparison,
) -> list[str]:
    lines: list[str] = [
        "## Pruning Mode Comparison",
        "",
        f"selected_mode: {comparison.selected_mode}",
        "",
        "| mode | strategy_id | rules | prompt_tokens | score | delta_vs_support_count | evaluated |",  # noqa: E501
        "|---|---|---:|---:|---:|---:|---|",  # noqa: E501
    ]
    for row in comparison.rows:
        score_str = f"{row.score:.6f}" if row.score is not None else "null"
        delta_str = (
            f"{row.delta_vs_support_count:+.6f}"
            if row.delta_vs_support_count is not None
            else "null"
        )
        lines.append(
            f"| {row.mode} | {row.strategy_id} | {row.rule_count} | {row.prompt_tokens} "
            f"| {score_str} | {delta_str} | {row.evaluated} |"
        )
    lines.append("")
    return lines


def _render_regressed_labels_section(rows: list[RegressedLabelRow]) -> list[str]:
    lines: list[str] = ["## Regressed Labels (recall_delta < 0)", ""]
    if not rows:
        lines.append("No regressed labels detected.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| label | support | baseline_recall | candidate_recall | recall_delta | baseline_f1 | "
            "candidate_f1 | f1_delta | new_false_negatives | top_predicted_wrong_labels | "
            "example_case_ids |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.label} | {row.support} | {row.baseline_recall:.6f} | "
            f"{row.candidate_recall:.6f} | "
            f"{row.recall_delta:.6f} | {row.baseline_f1:.6f} | {row.candidate_f1:.6f} | "
            f"{row.f1_delta:.6f} | {row.new_false_negatives} | "
            f"{', '.join(row.top_predicted_wrong_labels)} | {', '.join(row.example_case_ids)} |"
        )
    lines.append("")
    return lines


def _render_student_matrix_section(
    student_results: dict[str, StudentEvalSummary],
) -> list[str]:
    """Render a strategy × student metric matrix table."""
    if not student_results:
        return []
    lines: list[str] = [
        "## Student Evaluation Matrix",
        "",
        "| student_id | macro_f1 | accuracy | malformed_rate | cost_usd |",
        "|---|---:|---:|---:|---:|",
    ]
    for student_id, summary in sorted(student_results.items()):
        macro_f1_str = f"{summary.macro_f1:.6f}" if summary.macro_f1 is not None else "null"
        accuracy_str = f"{summary.accuracy:.6f}" if summary.accuracy is not None else "null"
        cost_str = f"{summary.cost_usd:.4f}" if summary.cost_usd is not None else "null"
        lines.append(
            f"| {student_id} | {macro_f1_str} | {accuracy_str} "
            f"| {summary.malformed_rate:.4f} | {cost_str} |"
        )
    lines.append("")
    return lines


def _render_top_confusions_section(rows: list[TopConfusionRow]) -> list[str]:
    lines: list[str] = ["## Top Confusions", ""]
    if not rows:
        lines.append("No non-diagonal confusions found.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| rank | actual_label | predicted_label | count | example_case_ids |",
            "|---:|---|---|---:|---|",
        ]
    )
    for index, row in enumerate(rows[:20], start=1):
        lines.append(
            "| "
            f"{index} | {row.actual_label} | {row.predicted_label} | {row.count} | "
            f"{', '.join(row.example_case_ids)} |"
        )
    lines.append("")
    return lines


def render_summary_markdown(
    manifest: BenchmarkManifest,
    dataset_manifest: DatasetManifest,
    comparison: BenchmarkStrategyComparison,
    *,
    reproduction_command: str,
) -> str:
    """Render benchmark summary markdown with CI, paired outcomes, and regressions."""
    paired_summary = comparison.paired_comparison
    candidate_eval = comparison.rulekiln_eval

    result_lines: list[str] = [
        "# BANKING77 Benchmark Summary",
        "",
        f"- run_id: {manifest.run_id}",
        f"- benchmark: {manifest.benchmark_name}",
        f"- profile: {dataset_manifest.profile}",
        f"- seed: {manifest.seed}",
        f"- dataset: {manifest.dataset_name}",
        f"- dataset_revision: {manifest.dataset_revision or 'unknown'}",
        "",
        "## Results",
        "",
        f"- primary_metric: {comparison.primary_metric}",
        f"- baseline_score: {comparison.baseline_score:.6f}",
        f"- candidate_score: {comparison.rulekiln_score:.6f}",
        f"- delta_vs_baseline: {comparison.delta_vs_baseline:.6f}",
        f"- selected_strategy: {comparison.selected_strategy}",
        f"- macro_f1: {candidate_eval.macro_f1 if candidate_eval.macro_f1 is not None else 'null'}",
        f"- macro_f1_ci_95: {_format_ci(candidate_eval.macro_f1_ci_95)}",
        f"- accuracy: {candidate_eval.accuracy if candidate_eval.accuracy is not None else 'null'}",
        f"- accuracy_ci_95: {_format_ci(candidate_eval.accuracy_ci_95)}",
    ]

    if paired_summary is not None:
        net_fix_rate_value: float | str = (
            paired_summary.net_fix_rate if paired_summary.net_fix_rate is not None else "null"
        )
        result_lines.extend(
            [
                f"- fixed_count: {paired_summary.fixed_count}",
                f"- broken_count: {paired_summary.broken_count}",
                f"- unchanged_correct_count: {paired_summary.unchanged_correct_count}",
                f"- unchanged_wrong_count: {paired_summary.unchanged_wrong_count}",
                f"- net_fix_rate: {net_fix_rate_value}",
                f"- overall_net_fix_rate: {paired_summary.overall_net_fix_rate}",
            ]
        )

    if comparison.pruning_mode_comparison is not None:
        result_lines.extend(
            ["", *_render_pruning_mode_comparison_section(comparison.pruning_mode_comparison)]
        )

    if comparison.student_results:
        result_lines.extend(["", *_render_student_matrix_section(comparison.student_results)])

    result_lines.extend(["", *(_render_regressed_labels_section(candidate_eval.regressed_labels))])
    result_lines.extend(_render_top_confusions_section(candidate_eval.top_confusions))
    result_lines.extend(
        [
            "## Caveats",
            "",
            "- Confidence intervals use deterministic bootstrap sampling and are sensitive "
            "to split size.",
            "- Macro F1 treats all labels equally and may over-emphasize rare-label variance.",
            "",
            "## Reproduction",
            "",
            "```bash",
            reproduction_command,
            "```",
            "",
            "## Benchmark Manifest Snapshot",
            "",
            f"- Git commit: {manifest.git_commit}",
            f"- RuleKiln version: {manifest.rulekiln_version}",
            f"- Python version: {manifest.python_version}",
            f"- Teacher model: {manifest.teacher_model}",
            f"- Student model: {manifest.student_model}",
            f"- Embedding model: {manifest.embedding_model}",
            "",
            "```json",
            json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(result_lines)


def write_summary_markdown(
    path: Path,
    manifest: BenchmarkManifest,
    dataset_manifest: DatasetManifest,
    comparison: BenchmarkStrategyComparison,
    *,
    reproduction_command: str,
) -> Path:
    """Write summary markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_summary_markdown(
        manifest,
        dataset_manifest,
        comparison,
        reproduction_command=reproduction_command,
    )
    path.write_text(content, encoding="utf-8")
    return path


def render_readme_snapshot_block(
    manifest: BenchmarkManifest,
    dataset_manifest: DatasetManifest,
    comparison: BenchmarkStrategyComparison,
) -> str:
    """Render the README snapshot block content."""
    return "\n".join(
        [
            "## Latest Reproducible Benchmark Snapshot",
            "",
            f"- Run ID: {manifest.run_id}",
            f"- Profile: {dataset_manifest.profile}",
            f"- Seed: {manifest.seed}",
            f"- Baseline ({comparison.primary_metric}): {comparison.baseline_score:.6f}",
            f"- RuleKiln ({comparison.primary_metric}): {comparison.rulekiln_score:.6f}",
            f"- Delta: {comparison.delta_vs_baseline:.6f}",
            f"- Selected strategy: {comparison.selected_strategy}",
            f"- Manifest git commit: {manifest.git_commit}",
            "",
        ]
    )


def update_readme_snapshot(
    readme_path: Path,
    manifest: BenchmarkManifest,
    dataset_manifest: DatasetManifest,
    comparison: BenchmarkStrategyComparison,
) -> Path:
    """Update README snapshot block with latest benchmark run metadata."""
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")

    current = readme_path.read_text(encoding="utf-8")
    snapshot = render_readme_snapshot_block(manifest, dataset_manifest, comparison)
    wrapped = f"{_SNAPSHOT_START}\n{snapshot}\n{_SNAPSHOT_END}"

    pattern = re.compile(
        rf"{re.escape(_SNAPSHOT_START)}.*?{re.escape(_SNAPSHOT_END)}",
        flags=re.DOTALL,
    )

    if pattern.search(current):
        updated = pattern.sub(wrapped, current)
    else:
        updated = current.rstrip() + "\n\n" + wrapped + "\n"

    readme_path.write_text(updated, encoding="utf-8")
    return readme_path
