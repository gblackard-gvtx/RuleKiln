"""DBOS stage-level workflow orchestration for RuleKiln.

This module defines a deterministic workflow with explicit stage steps:

- validate_project
- compile_prompts
- evaluate_baseline
- evaluate_dbscan
- evaluate_hdbscan
- aggregate_evaluation_report

It intentionally keeps existing RuleKiln stage markers and persistence checks
as idempotency guards during incremental migration.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.jobs import update_job_status
from rulekiln.db.session import get_session_factory
from rulekiln.schemas.job import DistillationRequest
from rulekiln.workers.distillation_worker import PipelineStage, run_pipeline_phase

try:
    from dbos import DBOS, SetWorkflowID
except Exception:  # pragma: no cover - DBOS is optional in some local test envs
    DBOS = None  # type: ignore[assignment]
    SetWorkflowID = None  # type: ignore[assignment]

def _workflow_decorator(name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    if DBOS is None:

        def _noop(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return _noop
    return DBOS.workflow(name=name)


def _step_decorator(name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    if DBOS is None:

        def _noop(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return _noop
    return DBOS.step(name=name, retries_allowed=False)


async def _await_if_needed[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _payload_from_json(payload_json: dict[str, object]) -> DistillationRequest:
    return DistillationRequest.model_validate(payload_json)


async def _run_validate_project_phase(job_id: str, payload_json: dict[str, object]) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="validate_project")


async def _run_compile_prompts_phase(job_id: str, payload_json: dict[str, object]) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")


async def _run_evaluate_baseline_phase(job_id: str, payload_json: dict[str, object]) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_baseline")


async def _run_evaluate_dbscan_phase(job_id: str, payload_json: dict[str, object]) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_dbscan")


async def _run_evaluate_hdbscan_phase(job_id: str, payload_json: dict[str, object]) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_hdbscan")


async def _run_aggregate_phase(job_id: str, payload_json: dict[str, object]) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="aggregate_evaluation_report")


@_step_decorator("validate_project")
async def _validate_project_step(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_validate_project_phase(job_id, payload_json)


@_step_decorator("compile_prompts")
async def _compile_prompts_step(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_compile_prompts_phase(job_id, payload_json)


@_step_decorator("evaluate_baseline")
async def _evaluate_baseline_step(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_evaluate_baseline_phase(job_id, payload_json)


@_step_decorator("evaluate_dbscan")
async def _evaluate_dbscan_step(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_evaluate_dbscan_phase(job_id, payload_json)


@_step_decorator("evaluate_hdbscan")
async def _evaluate_hdbscan_step(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_evaluate_hdbscan_phase(job_id, payload_json)


@_step_decorator("aggregate_evaluation_report")
async def _aggregate_evaluation_report_step(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_aggregate_phase(job_id, payload_json)


async def _run_stage_sequence(job_id: str, payload_json: dict[str, object]) -> None:
    await _validate_project_step(job_id, payload_json)
    await _compile_prompts_step(job_id, payload_json)
    await _evaluate_baseline_step(job_id, payload_json)
    await _evaluate_dbscan_step(job_id, payload_json)
    await _evaluate_hdbscan_step(job_id, payload_json)
    await _aggregate_evaluation_report_step(job_id, payload_json)


@_workflow_decorator("rulekiln_stage_workflow")
async def _run_rulekiln_stage_workflow(job_id: str, payload_json: dict[str, object]) -> None:
    await _run_stage_sequence(job_id, payload_json)


def _workflow_id_for_job(job_id: str) -> str:
    return f"rulekiln-job-{job_id}"


async def _start_stage_workflow(
    workflow_id: str,
    job_id: str,
    payload_json: dict[str, object],
) -> None:
    assert DBOS is not None  # noqa: S101
    assert SetWorkflowID is not None  # noqa: S101
    with SetWorkflowID(workflow_id):
        handle = await _await_if_needed(
            DBOS.start_workflow_async(_run_rulekiln_stage_workflow, job_id, payload_json)
        )
    await _await_if_needed(handle.get_result())


async def run_dbos_stage_workflow(job_id: str, payload: DistillationRequest) -> None:
    """Start or resume the DBOS stage workflow for one RuleKiln job."""
    payload_json = payload.model_dump(mode="json")

    # Fallback for test/runtime environments where DBOS is intentionally absent.
    if DBOS is None or SetWorkflowID is None:
        await _run_stage_sequence(job_id, payload_json)
        return

    workflow_id = _workflow_id_for_job(job_id)
    existing_status = await _await_if_needed(DBOS.get_workflow_status_async(workflow_id))

    if existing_status is None:
        await _start_stage_workflow(workflow_id, job_id, payload_json)
        return

    status_value = str(getattr(existing_status, "status", "") or "").upper()
    if status_value in {"ERROR", "CANCELLED"}:
        # Resume can replay the last failure immediately when step outcomes are
        # already persisted. Deleting and starting fresh preserves stage-marker
        # idempotency while allowing patched logic to run.
        await _await_if_needed(DBOS.delete_workflow_async(workflow_id))
        await _start_stage_workflow(workflow_id, job_id, payload_json)
        return

    # For running/success/pending states, reusing the same workflow ID returns
    # the existing execution handle/result without duplicating completed steps.
    await _start_stage_workflow(workflow_id, job_id, payload_json)


async def run_dbos_spike_workflow(
    session: AsyncSession,
    job_id: str,
    payload: DistillationRequest,
) -> None:
    """Backward-compatible spike runner kept for older tests.

    This now executes the first-pass stage workflow directly without DBOS runtime.
    """
    _ = session  # preserve signature compatibility for existing callers
    payload_json = payload.model_dump(mode="json")
    await _run_validate_project_phase(job_id, payload_json)
    await _run_compile_prompts_phase(job_id, payload_json)
    await _run_evaluate_baseline_phase(job_id, payload_json)
    async with get_session_factory()() as new_session:
        await update_job_status(
            new_session, job_id, status="completed", stage=PipelineStage.COMPLETED
        )
