"""Unit tests for MLflow token/cost metric helpers."""

from __future__ import annotations

from rulekiln.integrations.mlflow_tracker import build_token_cost_metrics


def test_mlflow_logs_cost_metrics() -> None:
    summary: dict[str, object] = {
        "total_tokens": 5000,
        "total_input_tokens": 3000,
        "total_output_tokens": 2000,
        "estimated_total_cost_usd": 0.012,
        "teacher_cost_usd": 0.008,
        "student_cost_usd": 0.003,
        "embedding_cost_usd": 0.001,
        "judge_cost_usd": 0.0,
        "total_model_calls": 10,
    }
    metrics = build_token_cost_metrics(summary)

    assert metrics["cost.total_usd"] == 0.012
    assert metrics["cost.teacher_usd"] == 0.008
    assert metrics["cost.student_usd"] == 0.003
    assert metrics["cost.embedding_usd"] == 0.001
    assert metrics["cost.judge_usd"] == 0.0


def test_mlflow_logs_token_metrics() -> None:
    summary: dict[str, object] = {
        "total_tokens": 5000,
        "total_input_tokens": 3000,
        "total_output_tokens": 2000,
        "estimated_total_cost_usd": 0.0,
        "teacher_cost_usd": 0.0,
        "student_cost_usd": 0.0,
        "embedding_cost_usd": 0.0,
        "judge_cost_usd": 0.0,
        "total_model_calls": 4,
    }
    metrics = build_token_cost_metrics(summary)

    assert metrics["tokens.total"] == 5000
    assert metrics["tokens.input"] == 3000
    assert metrics["tokens.output"] == 2000
    assert metrics["model_calls.total"] == 4
