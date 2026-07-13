"""Artifact writer: writes job outputs to the canonical directory layout (C007)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from rulekiln.artifacts.manifest import ManifestEntry
from rulekiln.pipeline.statistics import PairedComparisonArtifacts
from rulekiln.schemas.pipeline import (
    EvalResult,
    PerLabelMetricsRow,
    RuleAblationArtifact,
    RuleProvenanceArtifact,
    StrategyComparison,
    SynthesizedRuleSchema,
    TopConfusionRow,
)
from rulekiln.schemas.promotion import (
    ArtifactHashes,
    PromotionStatus,
    ProvenanceValidationReport,
)
from rulekiln.schemas.provenance import (
    ClusterAssignmentRecord,
    ClusteringProvenance,
    ClusterSummary,
)
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask


def _clustering_dir(root: Path, algorithm: str) -> Path:
    path = root / "outputs" / "clustering" / algorithm
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_clustering_provenance(
    root: Path, algorithm: str, provenance: ClusteringProvenance
) -> Path:
    """Write outputs/clustering/{algorithm}/clustering_provenance.json."""
    path = _clustering_dir(root, algorithm) / "clustering_provenance.json"
    path.write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_cluster_assignments(
    root: Path, algorithm: str, records: list[ClusterAssignmentRecord]
) -> Path:
    """Write outputs/clustering/{algorithm}/cluster_assignments.jsonl (deterministic order)."""
    path = _clustering_dir(root, algorithm) / "cluster_assignments.jsonl"
    lines = [json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cluster_summary(root: Path, algorithm: str, summary: ClusterSummary) -> Path:
    """Write outputs/clustering/{algorithm}/cluster_summary.json."""
    path = _clustering_dir(root, algorithm) / "cluster_summary.json"
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_clustering_synthesized_rules(
    root: Path, algorithm: str, rules: list[SynthesizedRuleSchema]
) -> Path:
    """Write outputs/clustering/{algorithm}/synthesized_rules.json."""
    path = _clustering_dir(root, algorithm) / "synthesized_rules.json"
    path.write_text(
        json.dumps([r.model_dump(mode="json") for r in rules], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_artifact_hashes(root: Path, hashes: ArtifactHashes) -> Path:
    """Write metadata/hashes.json."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "hashes.json"
    path.write_text(
        json.dumps(hashes.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_provenance_validation(root: Path, report: ProvenanceValidationReport) -> Path:
    """Write metadata/provenance_validation.json."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "provenance_validation.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_promotion_status(root: Path, status: PromotionStatus) -> Path:
    """Write metadata/promotion_status.json."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "promotion_status.json"
    path.write_text(
        json.dumps(status.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def job_artifact_root(artifact_root: str, job_id: str) -> Path:
    return Path(artifact_root) / job_id


def write_task(root: Path, task: RuleKilnTask) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "task.yaml"
    path.write_text(
        yaml.safe_dump(task.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_cases_normalized(root: Path, cases: list[RuleKilnCase]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cases.normalized.jsonl"
    lines = [json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in cases]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_prompt(root: Path, strategy: str, system_prompt: str) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"distilled_prompt_{strategy}.md"
    path.write_text(system_prompt, encoding="utf-8")
    return path


def write_baseline_prompt(root: Path, system_prompt: str) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "baseline_prompt.md"
    path.write_text(system_prompt, encoding="utf-8")
    return path


def write_baseline_scaffold_prompt(root: Path, system_prompt: str) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "baseline_scaffold_prompt.md"
    path.write_text(system_prompt, encoding="utf-8")
    return path


def write_strategy_prompt(root: Path, strategy: str, system_prompt: str) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"{strategy}_prompt.md"
    path.write_text(system_prompt, encoding="utf-8")
    return path


def write_selected_prompt(root: Path, system_prompt: str) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "selected_distilled_prompt.md"
    path.write_text(system_prompt, encoding="utf-8")
    return path


def write_rules(root: Path, strategy: str, rules: list[SynthesizedRuleSchema]) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"rules_{strategy}.jsonl"
    lines = [json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in rules]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_eval_report(root: Path, comparison: StrategyComparison) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "eval_report.json"
    strategy_evals_exclude: dict[str, set[str]] = {
        strategy_name: {"case_results"} for strategy_name in comparison.strategy_evals
    }
    exclude_payload: dict[str, set[str] | dict[str, set[str]]] = {
        "dbscan_eval": {"case_results"},
        "hdbscan_eval": {"case_results"},
        "baseline_eval": {"case_results"},
    }
    if strategy_evals_exclude:
        exclude_payload["strategy_evals"] = strategy_evals_exclude
    report = comparison.model_dump(
        mode="json",
        exclude=exclude_payload,
    )
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_strategy_eval(root: Path, strategy: str, eval_result: EvalResult) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"{strategy}_eval.json"
    path.write_text(
        json.dumps(eval_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _debug_expected_label(case: RuleKilnCase | None) -> str | None:
    if case is None or case.expected is None:
        return None
    raw = case.expected.get("label") if isinstance(case.expected, dict) else case.expected
    if raw is None:
        return None
    return raw if isinstance(raw, str) else str(raw)


def _debug_predicted_label(
    actual_output: dict[str, str | int | float | bool | None] | str | None,
) -> str | None:
    if isinstance(actual_output, dict):
        raw = actual_output.get("label")
    elif isinstance(actual_output, str):
        raw = actual_output
    else:
        raw = None
    if raw is None:
        return None
    return raw if isinstance(raw, str) else str(raw)


def write_strategy_debug_artifacts(
    root: Path,
    strategy: str,
    eval_result: EvalResult,
    cases_by_id: dict[str, RuleKilnCase],
    compiled_prompt: str,
    prompt_hash: str,
    baseline_eval: EvalResult | None = None,
) -> list[Path]:
    """Write per-strategy debugging artifacts so malformed/regressed cases can be
    inspected directly from disk without querying the database.

    Writes, under outputs/debug/: {strategy}_raw_outputs.jsonl, _parsed_outputs.jsonl,
    _malformed_outputs.jsonl, _regressions_vs_baseline.jsonl, _prompt_preview.txt,
    _compiled_prompt.txt, _confusion_matrix.json, _classification_report.json.
    """
    debug_dir = root / "outputs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    raw_path = debug_dir / f"{strategy}_raw_outputs.jsonl"
    raw_lines = [
        json.dumps({"case_id": r.case_id, "raw_content": r.raw_content}, ensure_ascii=False)
        for r in eval_result.case_results
    ]
    raw_path.write_text("\n".join(raw_lines), encoding="utf-8")
    paths.append(raw_path)

    parsed_path = debug_dir / f"{strategy}_parsed_outputs.jsonl"
    parsed_lines = [
        json.dumps(
            {
                "case_id": r.case_id,
                "parsed_label": _debug_predicted_label(r.actual_output),
                "expected_label": _debug_expected_label(cases_by_id.get(r.case_id)),
                "passed": r.passed,
                "recovered": r.recovered,
            },
            ensure_ascii=False,
        )
        for r in eval_result.case_results
        if not r.malformed
    ]
    parsed_path.write_text("\n".join(parsed_lines), encoding="utf-8")
    paths.append(parsed_path)

    malformed_path = debug_dir / f"{strategy}_malformed_outputs.jsonl"
    malformed_lines = [
        json.dumps(
            {
                "case_id": r.case_id,
                "expected_label": _debug_expected_label(cases_by_id.get(r.case_id)),
                "raw_content": r.raw_content,
                "actual_output": r.actual_output,
                "parse_error": r.error,
                "recovered": r.recovered,
            },
            ensure_ascii=False,
        )
        for r in eval_result.case_results
        if r.malformed
    ]
    malformed_path.write_text("\n".join(malformed_lines), encoding="utf-8")
    paths.append(malformed_path)

    regressions_path = debug_dir / f"{strategy}_regressions_vs_baseline.jsonl"
    regression_lines: list[str] = []
    if baseline_eval is not None:
        baseline_by_id = {r.case_id: r for r in baseline_eval.case_results}
        for r in eval_result.case_results:
            baseline_result = baseline_by_id.get(r.case_id)
            if baseline_result is None or not baseline_result.passed or r.passed:
                continue
            regression_lines.append(
                json.dumps(
                    {
                        "case_id": r.case_id,
                        "expected_label": _debug_expected_label(cases_by_id.get(r.case_id)),
                        "baseline_raw_content": baseline_result.raw_content,
                        "baseline_actual_output": baseline_result.actual_output,
                        "candidate_raw_content": r.raw_content,
                        "candidate_actual_output": r.actual_output,
                        "candidate_malformed": r.malformed,
                        "candidate_error": r.error,
                    },
                    ensure_ascii=False,
                )
            )
    regressions_path.write_text("\n".join(regression_lines), encoding="utf-8")
    paths.append(regressions_path)

    preview_path = debug_dir / f"{strategy}_prompt_preview.txt"
    preview_path.write_text(compiled_prompt[:2000], encoding="utf-8")
    paths.append(preview_path)

    full_prompt_path = debug_dir / f"{strategy}_compiled_prompt.txt"
    full_prompt_path.write_text(
        f"# prompt_hash: {prompt_hash}\n\n{compiled_prompt}", encoding="utf-8"
    )
    paths.append(full_prompt_path)

    confusion_path = debug_dir / f"{strategy}_confusion_matrix.json"
    confusion_path.write_text(
        json.dumps(eval_result.confusion_matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths.append(confusion_path)

    report_path = debug_dir / f"{strategy}_classification_report.json"
    report_path.write_text(
        json.dumps(
            [row.model_dump(mode="json") for row in eval_result.per_label_metrics],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths.append(report_path)

    return paths


def write_strategy_comparison(root: Path, comparison: StrategyComparison) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "strategy_comparison.json"
    path.write_text(
        json.dumps(comparison.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_failure_jsonl(root: Path, category: str, entries: list[dict[str, object]]) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"failures_{category}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_promptfoo_yaml(root: Path, task: RuleKilnTask, system_prompt: str) -> Path:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    path = exports / "promptfoo.yaml"
    doc: dict[str, object] = {
        "description": task.task_name,
        "prompts": [{"label": "selected", "raw": system_prompt}],
        "providers": [],
        "tests": [],
    }
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def write_mlflow_run_id(root: Path, run_id: str) -> Path:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    path = exports / "mlflow_run_id.txt"
    path.write_text(run_id, encoding="utf-8")
    return path


def write_manifest(
    root: Path,
    artifact_paths: list[str],
    entries: list[ManifestEntry] | None = None,
) -> Path:
    """Write metadata/manifest.json.

    ``artifacts`` (the flat path list) is preserved for existing readers; ``entries``
    carries per-artifact metadata (type, schema version, sha256, size, timestamp,
    strategy) when provided.
    """
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "manifest.json"
    payload: dict[str, object] = {"artifacts": artifact_paths}
    if entries is not None:
        payload["schema_version"] = "rulekiln.run_manifest.v1"
        payload["entries"] = [entry.model_dump(mode="json") for entry in entries]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_token_cost_summary(root: Path, summary: dict[str, object]) -> Path:
    """Write token usage and cost summary to metadata/token_cost_summary.json."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "token_cost_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_confusion_matrix_csv(path: Path, confusion_matrix: dict[str, dict[str, int]]) -> Path:
    """Write strict confusion matrix CSV sorted by actual and predicted labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual_label", "predicted_label", "count"])
        for actual_label in sorted(confusion_matrix.keys()):
            predicted_counts = confusion_matrix[actual_label]
            for predicted_label in sorted(predicted_counts.keys()):
                writer.writerow([actual_label, predicted_label, predicted_counts[predicted_label]])
    return path


def write_per_label_metrics_csv(path: Path, rows: list[PerLabelMetricsRow]) -> Path:
    """Write strict per-label metrics CSV sorted by label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: row.label)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "support",
                "true_positive",
                "false_positive",
                "false_negative",
                "precision",
                "recall",
                "f1",
            ]
        )
        for row in sorted_rows:
            writer.writerow(
                [
                    row.label,
                    row.support,
                    row.true_positive,
                    row.false_positive,
                    row.false_negative,
                    f"{row.precision:.10f}",
                    f"{row.recall:.10f}",
                    f"{row.f1:.10f}",
                ]
            )
    return path


def write_top_confusions_markdown(
    path: Path,
    top_confusions: list[TopConfusionRow],
    *,
    baseline_strategy_id: str | None = None,
    candidate_strategy_id: str | None = None,
) -> Path:
    """Write top confusion markdown with optional comparison context."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Top Confusions", ""]
    if baseline_strategy_id and candidate_strategy_id:
        lines.append(f"- Baseline strategy: {baseline_strategy_id}")
        lines.append(f"- Candidate strategy: {candidate_strategy_id}")
        lines.append("")

    if not top_confusions:
        lines.append("No non-diagonal confusions found.")
    else:
        lines.extend(
            [
                "| rank | actual_label | predicted_label | count | example_case_ids |",
                "|---:|---|---|---:|---|",
            ]
        )
        for index, row in enumerate(top_confusions[:20], start=1):
            lines.append(
                "| "
                f"{index} | {row.actual_label} | {row.predicted_label} | {row.count} | "
                f"{', '.join(row.example_case_ids)} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_paired_comparison_artifacts(
    path: Path,
    paired_comparison: PairedComparisonArtifacts,
) -> list[Path]:
    """Write strict paired comparison JSONL files plus summary.json."""
    path.mkdir(parents=True, exist_ok=True)

    fixed_path = path / "fixed.jsonl"
    broken_path = path / "broken.jsonl"
    unchanged_path = path / "unchanged.jsonl"
    summary_path = path / "summary.json"

    fixed_payload = "\n".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False)
        for row in paired_comparison.fixed_examples
    )
    broken_payload = "\n".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False)
        for row in paired_comparison.broken_examples
    )
    unchanged_payload = "\n".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False)
        for row in paired_comparison.unchanged_examples
    )

    fixed_path.write_text(fixed_payload, encoding="utf-8")
    broken_path.write_text(broken_payload, encoding="utf-8")
    unchanged_path.write_text(unchanged_payload, encoding="utf-8")
    summary_path.write_text(
        json.dumps(paired_comparison.summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return [fixed_path, broken_path, unchanged_path, summary_path]


def write_rule_provenance_json(root: Path, artifact: RuleProvenanceArtifact) -> Path:
    """Write rule_provenance.json to outputs/."""
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "rule_provenance.json"
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_rule_provenance_markdown(root: Path, artifact: RuleProvenanceArtifact) -> Path:
    """Write rule_provenance.md to outputs/ with a human-readable table."""
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "rule_provenance.md"

    lines: list[str] = [
        f"# Rule Provenance — {artifact.strategy_id}",
        "",
        f"job_id: {artifact.job_id}",
        "",
        "| rule_id | topic | support | attribution | ablation | flags |",
        "|---|---|---:|---|---|---|",
    ]
    for rec in artifact.rules:
        flags: list[str] = []
        if rec.zero_validation_impact:
            flags.append("zero_validation_impact")
        if rec.regression_flag:
            flags.append("regression_flag")
        ablation_str = (
            f"{rec.ablation_classification} (Δ={rec.ablation_metric_delta:.4f})"
            if rec.ablation_classification is not None and rec.ablation_metric_delta is not None
            else rec.ablation_classification or "—"
        )
        lines.append(
            f"| {rec.rule_id} | {rec.topic} | {rec.support_count} "
            f"| {rec.attribution_method} | {ablation_str} | {', '.join(flags) or '—'} |"
        )
    lines.append("")

    for rec in artifact.rules:
        lines.extend(
            [
                f"## {rec.rule_id}: {rec.topic}",
                "",
                f"- support_count: {rec.support_count}",
                f"- support_ratio: {rec.support_ratio:.4f}",
                f"- cluster_id: {rec.cluster_id or '—'}",
                f"- attribution_method: {rec.attribution_method}",
                f"- source_case_ids: {', '.join(rec.source_case_ids[:10])}"
                + (" …" if len(rec.source_case_ids) > 10 else ""),
                f"- examples_fixed: {', '.join(rec.examples_fixed[:5])}"
                + (" …" if len(rec.examples_fixed) > 5 else ""),
                f"- examples_broken: {', '.join(rec.examples_broken[:5])}"
                + (" …" if len(rec.examples_broken) > 5 else ""),
            ]
        )
        if rec.ablation_classification is not None:
            lines.extend(
                [
                    f"- ablation_classification: {rec.ablation_classification}",
                    f"- ablation_metric_delta: {rec.ablation_metric_delta}",
                    f"- ablation_changed_cases: {rec.ablation_changed_cases}",
                ]
            )
        if rec.notes:
            lines.append(f"- notes: {'; '.join(rec.notes)}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_rule_ablation_json(root: Path, artifact: RuleAblationArtifact) -> Path:
    """Write rule_ablation.json to outputs/."""
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "rule_ablation.json"
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
