"""Writers for benchmark artifacts and snapshots."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from rulekiln.benchmarks.schemas import (
    BenchmarkManifest,
    BenchmarkStrategyComparison,
    DatasetManifest,
    PerLabelMetricRow,
)
from rulekiln.schemas.pipeline import EvalResult

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
    """Write confusion matrix rows using strict schema.

    Header: expected_label,predicted_label,count
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["expected_label", "predicted_label", "count"])

        expected_labels = sorted(eval_result.confusion_matrix.keys())
        for expected_label in expected_labels:
            predicted_map = eval_result.confusion_matrix.get(expected_label, {})
            for predicted_label in sorted(predicted_map.keys()):
                count = predicted_map[predicted_label]
                writer.writerow([expected_label, predicted_label, count])

    return path


def write_per_label_metrics_csv(path: Path, rows: list[PerLabelMetricRow]) -> Path:
    """Write per-label metrics rows using strict schema.

    Header: label,precision,recall,support,strategy
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (row.strategy, row.label))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "support", "strategy"])
        for row in sorted_rows:
            writer.writerow(
                [
                    row.label,
                    f"{row.precision:.10f}",
                    f"{row.recall:.10f}",
                    row.support,
                    row.strategy,
                ]
            )

    return path


def render_summary_markdown(
    manifest: BenchmarkManifest,
    dataset_manifest: DatasetManifest,
    comparison: BenchmarkStrategyComparison,
) -> str:
    """Render a benchmark summary markdown snapshot."""
    return "\n".join(
        [
            "# BANKING77 Benchmark Summary",
            "",
            f"- Run ID: {manifest.run_id}",
            f"- Benchmark: {manifest.benchmark_name}",
            f"- Profile: {dataset_manifest.profile}",
            f"- Seed: {manifest.seed}",
            f"- Dataset: {manifest.dataset_name}",
            f"- Dataset revision: {manifest.dataset_revision or 'unknown'}",
            "",
            "## Results",
            "",
            f"- Primary metric: {comparison.primary_metric}",
            f"- Baseline score: {comparison.baseline_score:.6f}",
            f"- RuleKiln score: {comparison.rulekiln_score:.6f}",
            f"- Delta vs baseline: {comparison.delta_vs_baseline:.6f}",
            f"- Selected strategy: {comparison.selected_strategy}",
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


def write_summary_markdown(
    path: Path,
    manifest: BenchmarkManifest,
    dataset_manifest: DatasetManifest,
    comparison: BenchmarkStrategyComparison,
) -> Path:
    """Write summary markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_summary_markdown(manifest, dataset_manifest, comparison)
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
