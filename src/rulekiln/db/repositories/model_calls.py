"""Repository for ModelCallEvent persistence."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import DistillationJob, ModelCallEvent
from rulekiln.schemas.usage import ModelCallRecord


def _record_to_event(record: ModelCallRecord, job_id: str) -> ModelCallEvent:
    usage = record.usage
    cost = record.cost

    return ModelCallEvent(
        id=str(record.id),
        job_id=job_id,
        stage=record.stage,
        role=record.role,
        provider_profile=record.provider_profile,
        provider=record.provider,
        model=record.model,
        student_id=record.student_id,
        strategy=record.strategy,
        case_id=record.case_id,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        usage_estimated=usage.estimated if usage else True,
        input_cost_usd=float(cost.input_cost_usd) if cost else 0.0,
        output_cost_usd=float(cost.output_cost_usd) if cost else 0.0,
        total_cost_usd=float(cost.total_cost_usd) if cost else 0.0,
        cost_estimated=cost.estimated if cost else True,
        pricing_source=cost.pricing_source if cost else None,
        latency_ms=record.latency_ms,
        status=record.status,
        error_type=record.error_type,
    )


async def bulk_insert_model_call_events(
    session: AsyncSession,
    job_id: str,
    records: list[ModelCallRecord],
) -> None:
    """Insert all model call records for a job into the DB."""
    events = [_record_to_event(r, job_id) for r in records]
    session.add_all(events)
    await session.flush()


async def update_job_usage_totals(
    session: AsyncSession,
    job_id: str,
    summary: dict[str, object],
) -> None:
    """Write aggregated usage summary back to the DistillationJob row."""
    stmt = (
        update(DistillationJob)
        .where(DistillationJob.id == job_id)  # type: ignore[arg-type]
        .values(
            total_input_tokens=summary.get("total_input_tokens", 0),
            total_output_tokens=summary.get("total_output_tokens", 0),
            total_tokens=summary.get("total_tokens", 0),
            estimated_total_cost_usd=summary.get("estimated_total_cost_usd"),
            teacher_cost_usd=summary.get("teacher_cost_usd"),
            student_cost_usd=summary.get("student_cost_usd"),
            embedding_cost_usd=summary.get("embedding_cost_usd"),
            judge_cost_usd=summary.get("judge_cost_usd"),
        )
    )
    await session.execute(stmt)
    await session.flush()
