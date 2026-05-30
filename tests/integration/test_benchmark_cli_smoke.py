"""Integration smoke test for the benchmark CLI."""

from __future__ import annotations

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
        run_root / "summary.md",
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

    summary_content = (run_root / "summary.md").read_text(encoding="utf-8")
    assert "## Benchmark Manifest Snapshot" in summary_content
