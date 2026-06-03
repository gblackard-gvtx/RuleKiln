"""Repository for ModelCallEvent persistence."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import DistillationJob, ModelCallEvent
from rulekiln.schemas.usage import ModelCallRecord

type RoleUsageBucket = dict[str, int | float]


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
        idempotency_key=record.idempotency_key,
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
    if not records:
        return

    events = [_record_to_event(r, job_id) for r in records]
    deduped_events = _dedupe_events_by_key(events)
    existing_keys = await _get_existing_idempotency_keys(session, deduped_events)
    insertable_events = [
        event
        for event in deduped_events
        if event.idempotency_key is None or event.idempotency_key not in existing_keys
    ]
    if not insertable_events:
        return

    session.add_all(insertable_events)
    await session.flush()


def _dedupe_events_by_key(events: list[ModelCallEvent]) -> list[ModelCallEvent]:
    seen_keys: set[str] = set()
    deduped: list[ModelCallEvent] = []
    for event in events:
        key = event.idempotency_key
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        deduped.append(event)
    return deduped


async def _get_existing_idempotency_keys(
    session: AsyncSession,
    events: list[ModelCallEvent],
) -> set[str]:
    keys = {event.idempotency_key for event in events if event.idempotency_key is not None}
    if not keys:
        return set()

    result = await session.execute(
        select(ModelCallEvent.idempotency_key).where(ModelCallEvent.idempotency_key.in_(keys))
    )
    return {key for key in result.scalars().all() if key is not None}


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


async def summarize_model_call_events(
    session: AsyncSession,
    job_id: str,
) -> dict[str, object]:
    """Aggregate persisted model_call_events for one job into token/cost totals."""
    result = await session.execute(select(ModelCallEvent).where(ModelCallEvent.job_id == job_id))
    events = list(result.scalars().all())

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = Decimal("0")
    has_estimated_usage = False

    by_role: dict[str, RoleUsageBucket] = {}

    for event in events:
        input_tokens = event.input_tokens or 0
        output_tokens = event.output_tokens or 0
        token_total = event.total_tokens or (input_tokens + output_tokens)
        cost = Decimal(str(event.total_cost_usd or 0))

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += token_total
        total_cost += cost
        has_estimated_usage = has_estimated_usage or event.usage_estimated or event.cost_estimated

        role = event.role
        if role not in by_role:
            by_role[role] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "call_count": 0,
            }
        bucket = by_role[role]
        bucket["input_tokens"] = int(bucket["input_tokens"]) + input_tokens
        bucket["output_tokens"] = int(bucket["output_tokens"]) + output_tokens
        bucket["total_tokens"] = int(bucket["total_tokens"]) + token_total
        bucket["cost_usd"] = float(bucket["cost_usd"]) + float(cost)
        bucket["call_count"] = int(bucket["call_count"]) + 1

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost_usd": float(total_cost),
        "teacher_cost_usd": float(by_role.get("teacher", {}).get("cost_usd", 0.0)),
        "student_cost_usd": float(by_role.get("student", {}).get("cost_usd", 0.0)),
        "embedding_cost_usd": float(by_role.get("embedding", {}).get("cost_usd", 0.0)),
        "judge_cost_usd": float(by_role.get("judge", {}).get("cost_usd", 0.0)),
        "has_estimated_usage": has_estimated_usage,
        "total_model_calls": len(events),
        "by_role": by_role,
    }
