"""Artifact writer: writes job outputs to the canonical directory layout (C007)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rulekiln.importers.column_mapping import CsvImportMapping, CsvImportPreview
from rulekiln.schemas.pipeline import StrategyComparison, SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask


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
    report = comparison.model_dump(
        mode="json",
        exclude={
            "dbscan_eval": {"case_results"},
            "hdbscan_eval": {"case_results"},
            "baseline_eval": {"case_results"},
        },
    )
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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


def write_manifest(root: Path, artifact_paths: list[str]) -> Path:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "manifest.json"
    path.write_text(
        json.dumps({"artifacts": artifact_paths}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_import_mapping(root: Path, mapping: CsvImportMapping) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "import_mapping.yaml"
    path.write_text(
        yaml.safe_dump(mapping.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_import_preview(root: Path, preview: CsvImportPreview) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "import_preview.json"
    path.write_text(
        json.dumps(preview.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_import_source_csv(root: Path, content: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source.csv"
    path.write_bytes(content)
    return path


def write_validation_report(root: Path, errors: list[str], warnings: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "validation_report.json"
    path.write_text(
        json.dumps({"errors": errors, "warnings": warnings}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_cases_jsonl(root: Path, cases: list[RuleKilnCase]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cases.jsonl"
    lines = [json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in cases]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
