"""Integration smoke test for the benchmark CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rulekiln.benchmarks.cli import main


def _write_fixture(path: Path) -> None:
    labels = [
        "card_arrival",
        "top_up_failed",
        "declined_transfer",
        "cash_withdrawal_charge",
        "exchange_rate",
        "lost_or_stolen_card",
    ]

    lines: list[str] = []
    case_index = 0
    for label in labels:
        for sample_index in range(30):
            payload = {
                "schema_version": "rulekiln.case.v1",
                "id": f"fixture_{case_index:06d}",
                "split": "train",
                "task_mode": "classification",
                "input": {
                    "utterance": f"cli sample {sample_index} about {label}",
                },
                "expected": {"label": label},
                "evaluation": {
                    "assertions": [
                        {
                            "type": "must_equal",
                            "path": "$.label",
                            "value": label,
                            "weight": 1.0,
                        }
                    ]
                },
                "metadata": {"source": "fixture"},
                "weight": 1.0,
            }
            lines.append(json.dumps(payload))
            case_index += 1

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_benchmark_cli_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    fixture_path = tmp_path / "cases.sample.jsonl"
    _write_fixture(fixture_path)

    artifact_root = tmp_path / "benchmark_runs"
    run_id = "cli_smoke"

    exit_code = main(
        [
            "banking77",
            "--profile",
            "smoke",
            "--seed",
            "111",
            "--run-id",
            run_id,
            "--dataset-source",
            "fixture",
            "--fixture-path",
            str(fixture_path),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert exit_code == 0

    run_root = artifact_root / "banking77" / run_id
    required_files = [
        run_root / "benchmark_manifest.json",
        run_root / "dataset_manifest.json",
        run_root / "baseline_eval.json",
        run_root / "rulekiln_eval.json",
        run_root / "strategy_comparison.json",
        run_root / "confusion_matrix.csv",
        run_root / "per_label_metrics.csv",
        run_root / "top_confusions.md",
        run_root / "summary.md",
        run_root / "paired_comparison/fixed.jsonl",
        run_root / "paired_comparison/broken.jsonl",
        run_root / "paired_comparison/unchanged.jsonl",
        run_root / "paired_comparison/summary.json",
        run_root / "splits/train_ids.txt",
        run_root / "splits/validation_ids.txt",
        run_root / "splits/test_ids.txt",
    ]

    for artifact_path in required_files:
        assert artifact_path.exists(), f"Missing artifact: {artifact_path}"

    manifest = json.loads((run_root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    required_fields = {
        "benchmark_name",
        "run_id",
        "created_at",
        "git_commit",
        "rulekiln_version",
        "python_version",
        "dataset_name",
        "dataset_revision",
        "seed",
        "teacher_model",
        "student_model",
        "embedding_model",
        "strategy_names",
        "prompt_hashes",
        "case_counts",
        "cost_summary",
    }
    assert required_fields.issubset(set(manifest.keys()))

    rulekiln_eval = json.loads((run_root / "rulekiln_eval.json").read_text(encoding="utf-8"))
    accuracy_ci = rulekiln_eval.get("accuracy_ci_95")
    macro_f1_ci = rulekiln_eval.get("macro_f1_ci_95")
    assert isinstance(accuracy_ci, dict)
    assert isinstance(macro_f1_ci, dict)
    assert {"low", "high", "method", "iterations", "seed"}.issubset(accuracy_ci.keys())
    assert {"low", "high", "method", "iterations", "seed"}.issubset(macro_f1_ci.keys())

    summary_content = (run_root / "summary.md").read_text(encoding="utf-8")
    assert "- macro_f1_ci_95:" in summary_content
    assert "- accuracy_ci_95:" in summary_content
    assert "## Regressed Labels" in summary_content
    assert "## Top Confusions" in summary_content
    assert "## Reproduction" in summary_content
    assert "## Benchmark Manifest Snapshot" in summary_content

    with (run_root / "confusion_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["actual_label", "predicted_label", "count"]
    assert all(len(row) == 3 for row in rows[1:])
    sorted_pairs = sorted((row[0], row[1]) for row in rows[1:])
    assert [(row[0], row[1]) for row in rows[1:]] == sorted_pairs

    with (run_root / "per_label_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "label",
        "support",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]
    labels = [row[0] for row in rows[1:]]
    assert labels == sorted(labels)

    paired_summary = json.loads(
        (run_root / "paired_comparison/summary.json").read_text(encoding="utf-8")
    )
    fixed_count = len(
        [
            line
            for line in (run_root / "paired_comparison/fixed.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    )
    broken_count = len(
        [
            line
            for line in (run_root / "paired_comparison/broken.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    )
    unchanged_rows = [
        line
        for line in (run_root / "paired_comparison/unchanged.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    unchanged_count = len(unchanged_rows)
    unchanged_payloads = [json.loads(line) for line in unchanged_rows]
    assert all(
        payload["unchanged_status"] in {"both_correct", "both_wrong"}
        for payload in unchanged_payloads
    )

    assert paired_summary["fixed_count"] == fixed_count
    assert paired_summary["broken_count"] == broken_count
    assert (
        paired_summary["unchanged_correct_count"] + paired_summary["unchanged_wrong_count"]
    ) == unchanged_count
