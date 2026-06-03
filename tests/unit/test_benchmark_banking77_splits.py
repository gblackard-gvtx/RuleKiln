"""Deterministic split tests for BANKING77 benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

from rulekiln.benchmarks.banking77 import run_banking77_benchmark


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
                    "utterance": f"example {sample_index} about {label}",
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


def _split_bytes(run_root: Path) -> tuple[bytes, bytes, bytes]:
    train_bytes = (run_root / "splits/train_ids.txt").read_bytes()
    validation_bytes = (run_root / "splits/validation_ids.txt").read_bytes()
    test_bytes = (run_root / "splits/test_ids.txt").read_bytes()
    return train_bytes, validation_bytes, test_bytes


def test_split_id_files_are_byte_identical_for_same_seed(tmp_path: Path) -> None:
    fixture_path = tmp_path / "cases.sample.jsonl"
    _write_fixture(fixture_path)

    artifact_root = tmp_path / "benchmark_runs"

    run_a = run_banking77_benchmark(
        profile="smoke",
        seed=101,
        run_id="same_seed_a",
        artifact_root=artifact_root,
        dataset_source="fixture",
        fixture_path=fixture_path,
    )
    run_b = run_banking77_benchmark(
        profile="smoke",
        seed=101,
        run_id="same_seed_b",
        artifact_root=artifact_root,
        dataset_source="fixture",
        fixture_path=fixture_path,
    )

    assert _split_bytes(run_a.run_root) == _split_bytes(run_b.run_root)


def test_split_id_files_change_when_seed_changes(tmp_path: Path) -> None:
    fixture_path = tmp_path / "cases.sample.jsonl"
    _write_fixture(fixture_path)

    artifact_root = tmp_path / "benchmark_runs"

    run_a = run_banking77_benchmark(
        profile="smoke",
        seed=101,
        run_id="seed_101",
        artifact_root=artifact_root,
        dataset_source="fixture",
        fixture_path=fixture_path,
    )
    run_b = run_banking77_benchmark(
        profile="smoke",
        seed=202,
        run_id="seed_202",
        artifact_root=artifact_root,
        dataset_source="fixture",
        fixture_path=fixture_path,
    )

    assert _split_bytes(run_a.run_root) != _split_bytes(run_b.run_root)
