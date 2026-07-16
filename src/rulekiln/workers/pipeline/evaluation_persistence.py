"""Evaluation DB persistence helpers: load, insert, upsert eval runs and case results."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.eval_case_results import (
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.db.repositories.jobs import get_eval_runs_for_job, insert_eval_run
from rulekiln.db.repositories.model_calls import bulk_insert_model_call_events
from rulekiln.providers.tracking import ModelCallCollector
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import RuleKilnCase
from rulekiln.workers.pipeline.db_mappers import (
    _build_eval_case_upsert_payload,
    _eval_case_record_to_schema,
    _eval_to_db,
)


async def _persist_collector_records(
    session: AsyncSession,
    job_id: str,
    collector: ModelCallCollector,
) -> None:
    if not collector.records:
        return
    await bulk_insert_model_call_events(session, job_id, collector.records)
    await session.commit()


async def _load_eval_result_from_db(
    session: AsyncSession,
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
) -> EvalResult | None:
    eval_runs = await get_eval_runs_for_job(session, job_id)
    run = next((row for row in eval_runs if row.strategy == strategy and row.split == split), None)
    if run is None:
        return None

    case_rows = await get_eval_case_results(
        session,
        job_id=job_id,
        student_id=student_id,
        strategy=strategy,
        split=split,
    )
    case_results = [_eval_case_record_to_schema(row) for row in case_rows]

    confusion_matrix_raw = run.confusion_matrix or {}
    confusion_matrix: dict[str, dict[str, int]] = {
        str(expected): {str(actual): int(count) for actual, count in actuals.items()}
        for expected, actuals in confusion_matrix_raw.items()
    }

    return EvalResult(
        strategy=run.strategy,
        model=run.model,
        split=run.split,
        metric_semantics_version=(
            run.metric_semantics_version if run.metric_semantics_version is not None else 1
        ),
        accuracy=run.accuracy,
        macro_f1=run.macro_f1,
        valid_output_accuracy=run.valid_output_accuracy,
        valid_output_macro_f1=run.valid_output_macro_f1,
        weighted_case_score=run.weighted_case_score,
        malformed_output_rate=run.malformed_output_rate or 0.0,
        per_outcome_precision=dict(run.per_outcome_precision or {}),
        per_outcome_recall=dict(run.per_outcome_recall or {}),
        confusion_matrix=confusion_matrix,
        case_results=case_results,
    )


async def _insert_eval_run_if_missing(
    session: AsyncSession,
    *,
    job_id: str,
    split: str,
    strategy: str,
    result: EvalResult,
) -> None:
    existing_eval_runs = await get_eval_runs_for_job(session, job_id)
    has_strategy_eval = any(
        run.strategy == strategy and run.split == split for run in existing_eval_runs
    )
    if not has_strategy_eval:
        await insert_eval_run(session, _eval_to_db(job_id, None, result))


async def _persist_case_results(
    session: AsyncSession,
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
    case_by_id: dict[str, RuleKilnCase],
    case_results: list[CaseEvalResult],
) -> None:
    for case_result in case_results:
        case = case_by_id.get(case_result.case_id)
        if case is None:
            continue
        payload_row = _build_eval_case_upsert_payload(
            job_id=job_id,
            student_id=student_id,
            strategy=strategy,
            split=split,
            case=case,
            result=case_result,
        )
        await upsert_eval_case_result(session, payload_row)
