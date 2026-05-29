"""Unit tests for ModelUsageAggregator."""

from __future__ import annotations

from decimal import Decimal

import pytest

from rulekiln.schemas.usage import (
    ModelCallCost,
    ModelCallRecord,
    ModelUsage,
)
from rulekiln.usage.aggregator import ModelUsageAggregator


def _make_usage(input_t: int, output_t: int, estimated: bool = False) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        total_tokens=input_t + output_t,
        estimated=estimated,
    )


def _make_cost(input_usd: str, output_usd: str, estimated: bool = False) -> ModelCallCost:
    inp = Decimal(input_usd)
    out = Decimal(output_usd)
    return ModelCallCost(
        input_cost_usd=inp,
        output_cost_usd=out,
        total_cost_usd=inp + out,
        estimated=estimated,
        pricing_source="config",
    )


def _make_record(
    role: str,
    input_t: int,
    output_t: int,
    cost_input: str = "0.001",
    cost_output: str = "0.002",
    status: str = "success",
    stage: str = "extracting_rules",
    estimated_usage: bool = False,
) -> ModelCallRecord:
    return ModelCallRecord(
        job_id="job-1",
        stage=stage,
        role=role,  # type: ignore[arg-type]
        provider_profile="default",
        provider="openai",
        model="gpt-4o-mini",
        usage=_make_usage(input_t, output_t, estimated_usage),
        cost=_make_cost(cost_input, cost_output),
        latency_ms=100,
        status=status,  # type: ignore[arg-type]
    )


def test_job_usage_summary_aggregates_by_role() -> None:
    records = [
        _make_record("teacher", 1000, 500),
        _make_record("teacher", 2000, 800),
        _make_record("student", 500, 200),
        _make_record("embedding", 300, 0),
    ]
    aggregator = ModelUsageAggregator()
    summary = aggregator.aggregate(records)

    by_role = summary["by_role"]
    assert isinstance(by_role, dict)
    assert by_role["teacher"]["input_tokens"] == 3000  # type: ignore[index]
    assert by_role["teacher"]["output_tokens"] == 1300  # type: ignore[index]
    assert by_role["student"]["input_tokens"] == 500  # type: ignore[index]
    assert by_role["embedding"]["input_tokens"] == 300  # type: ignore[index]


def test_job_usage_summary_totals() -> None:
    records = [
        _make_record("teacher", 1000, 500, "0.001", "0.002"),
        _make_record("student", 500, 200, "0.0005", "0.001"),
    ]
    aggregator = ModelUsageAggregator()
    summary = aggregator.aggregate(records)

    assert summary["total_input_tokens"] == 1500
    assert summary["total_output_tokens"] == 700
    assert summary["total_tokens"] == 2200
    assert summary["total_model_calls"] == 2


def test_job_usage_summary_cost_by_role() -> None:
    records = [
        _make_record("teacher", 1000, 500, "0.001", "0.002"),
        _make_record("student", 500, 200, "0.0005", "0.001"),
    ]
    aggregator = ModelUsageAggregator()
    summary = aggregator.aggregate(records)

    assert summary["teacher_cost_usd"] == pytest.approx(0.003, rel=1e-6)
    assert summary["student_cost_usd"] == pytest.approx(0.0015, rel=1e-6)
    assert summary["embedding_cost_usd"] == pytest.approx(0.0, rel=1e-6)
    assert summary["judge_cost_usd"] == pytest.approx(0.0, rel=1e-6)


def test_token_cost_summary_has_estimated_flag() -> None:
    records = [
        _make_record("teacher", 1000, 500),
        _make_record("student", 300, 100, estimated_usage=True),
    ]
    aggregator = ModelUsageAggregator()
    summary = aggregator.aggregate(records)
    assert summary["has_estimated_usage"] is True


def test_estimated_false_when_all_exact() -> None:
    records = [
        _make_record("teacher", 1000, 500),
        _make_record("student", 300, 100),
    ]
    aggregator = ModelUsageAggregator()
    summary = aggregator.aggregate(records)
    assert summary["has_estimated_usage"] is False


def test_empty_records_produces_zeroes() -> None:
    aggregator = ModelUsageAggregator()
    summary = aggregator.aggregate([])
    assert summary["total_tokens"] == 0
    assert summary["total_model_calls"] == 0
    assert summary["estimated_total_cost_usd"] == pytest.approx(0.0)
