"""Unit tests for MLflow token/cost metric helpers."""

from __future__ import annotations

from rulekiln.integrations.mlflow_tracker import (
    build_demo_eval_metrics,
    build_demo_params,
    build_token_cost_metrics,
)


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


def test_mlflow_builds_minimum_demo_params() -> None:
    params = build_demo_params(
        task_id="task-1",
        task_mode="classification",
        dataset="cases_sha256:abc",
        teacher_provider="openai",
        teacher_model="gpt-4.1",
        student_provider="openai",
        student_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        selected_strategy="hdbscan",
        primary_metric="macro_f1",
    )

    assert params["task_id"] == "task-1"
    assert params["task_mode"] == "classification"
    assert params["dataset"] == "cases_sha256:abc"
    assert params["teacher_provider"] == "openai"
    assert params["teacher_model"] == "gpt-4.1"
    assert params["student_provider"] == "openai"
    assert params["student_model"] == "gpt-4.1-mini"
    assert params["embedding_model"] == "text-embedding-3-small"
    assert params["selected_strategy"] == "hdbscan"
    assert params["primary_metric"] == "macro_f1"


def test_mlflow_builds_minimum_demo_eval_metrics() -> None:
    metrics = build_demo_eval_metrics(
        baseline_macro_f1=0.71,
        baseline_accuracy=0.76,
        baseline_malformed_output_rate=0.02,
        dbscan_macro_f1=0.75,
        dbscan_accuracy=0.79,
        dbscan_delta_vs_baseline=0.04,
        hdbscan_macro_f1=0.77,
        hdbscan_accuracy=0.8,
        hdbscan_delta_vs_baseline=0.06,
        selected_primary_score=0.77,
        selected_delta_vs_baseline=0.06,
        selected_passed_quality_gates=True,
    )

    assert metrics["eval.baseline.macro_f1"] == 0.71
    assert metrics["eval.baseline.accuracy"] == 0.76
    assert metrics["eval.baseline.malformed_output_rate"] == 0.02
    assert metrics["eval.dbscan.macro_f1"] == 0.75
    assert metrics["eval.dbscan.accuracy"] == 0.79
    assert metrics["eval.dbscan.delta_vs_baseline"] == 0.04
    assert metrics["eval.hdbscan.macro_f1"] == 0.77
    assert metrics["eval.hdbscan.accuracy"] == 0.8
    assert metrics["eval.hdbscan.delta_vs_baseline"] == 0.06
    assert metrics["selected.primary_score"] == 0.77
    assert metrics["selected.delta_vs_baseline"] == 0.06
    assert metrics["selected.passed_quality_gates"] == 1.0
