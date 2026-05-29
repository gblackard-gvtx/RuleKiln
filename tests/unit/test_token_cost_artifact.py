"""Unit tests for write_token_cost_summary artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rulekiln.artifacts.writer import write_token_cost_summary


def test_token_cost_summary_artifact_is_written(tmp_path: Path) -> None:
    summary: dict[str, object] = {
        "total_input_tokens": 1000,
        "total_output_tokens": 500,
        "total_tokens": 1500,
        "estimated_total_cost_usd": 0.00123,
        "teacher_cost_usd": 0.001,
        "student_cost_usd": 0.0002,
        "embedding_cost_usd": 0.00003,
        "judge_cost_usd": 0.0,
        "has_estimated_usage": False,
        "total_model_calls": 7,
    }
    path = write_token_cost_summary(tmp_path, summary)
    assert path.exists()
    assert path.name == "token_cost_summary.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["total_tokens"] == 1500
    assert loaded["teacher_cost_usd"] == pytest.approx(0.001, rel=1e-6)
