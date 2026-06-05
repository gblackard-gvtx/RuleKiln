"""DBOS stage-level workflow orchestration for RuleKiln.

Batch extraction adds three new DBOS steps that bracket the durable poll loop:

  submit_extraction_batch_step   — calls run_pipeline_phase("extraction_batch_submit")
  poll_extraction_batch_step     — returns True when the provider batch is complete
  collect_extraction_batch_step  — calls run_pipeline_phase("extraction_batch_collect")

The poll loop lives in the workflow function (not a step) and uses DBOS.sleep for
durable sleep when the DBOS runtime is present, falling back to asyncio.sleep otherwise.

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

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import cast
from collections.abc import Awaitable, Callable, Coroutine
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.config.settings import get_settings
from rulekiln.db.repositories.jobs import (
    get_batch_job_by_stage,
    update_job_status,
)
from rulekiln.db.session import get_session_factory
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.contracts import BatchChatModelClient
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.schemas.job import DistillationRequest
from rulekiln.workers.distillation_worker import PipelineStage, run_pipeline_phase

try:
    from dbos import DBOS, SetWorkflowID
except Exception:  # pragma: no cover - DBOS is optional in some local test envs
    DBOS = None  # type: ignore[assignment]
    SetWorkflowID = None  # type: ignore[assignment]


type WorkflowFunc = Callable[[str, dict[str, object]], Coroutine[object, object, None]]
type WorkflowFuncBool = Callable[[str, dict[str, object]], Coroutine[object, object, bool]]


def _workflow_decorator(name: str) -> Callable[[WorkflowFunc], WorkflowFunc]:
    if DBOS is None:

        def _noop(func: WorkflowFunc) -> WorkflowFunc:
        def _noop(func: WorkflowFunc) -> WorkflowFunc:
            return func

        return _noop
    return cast(Callable[[WorkflowFunc], WorkflowFunc], DBOS.workflow(name=name))
    return cast(Callable[[WorkflowFunc], WorkflowFunc], DBOS.workflow(name=name))


def _step_decorator(name: str) -> Callable[[WorkflowFunc], WorkflowFunc]:
def _step_decorator(name: str) -> Callable[[WorkflowFunc], WorkflowFunc]:
    if DBOS is None:

        def _noop(func: WorkflowFunc) -> WorkflowFunc:
        def _noop(func: WorkflowFunc) -> WorkflowFunc:
            return func

        return _noop
    return cast(Callable[[WorkflowFunc], WorkflowFunc], DBOS.step(name=name, retries_allowed=False))


def _step_decorator_bool(name: str) -> Callable[[WorkflowFuncBool], WorkflowFuncBool]:
    if DBOS is None:

        def _noop_bool(func: WorkflowFuncBool) -> WorkflowFuncBool:
            return func

        return _noop_bool
    return cast(
        Callable[[WorkflowFuncBool], WorkflowFuncBool],
        DBOS.step(name=name, retries_allowed=False),
    )


async def _await_if_needed[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _payload_from_json(payload_json: dict[str, object]) -> DistillationRequest:
    return DistillationRequest.model_validate(payload_json)


def _extraction_batch_enabled(payload_json: dict[str, object]) -> bool:
    """Return True if the payload config requests batch extraction.

    This is a pure config read with no side effects, safe to call inside a
    DBOS workflow function (deterministic for the same payload).
    """
    try:
        payload = _payload_from_json(payload_json)
        tc = payload.teacher_config
        if tc is None:
            return False
        phase_cfg = tc.for_phase("instruction_extraction")
        if not phase_cfg.batch_enabled:
            return False
        settings = get_settings()
        profile = settings.provider_profiles.get(phase_cfg.provider)
        if profile is None or not profile.batch_enabled:
            return False
        config = resolve_provider_config(
            phase_cfg.provider,
            phase_cfg.model,
            role="teacher",
            settings=get_settings(),
        )
        client = get_chat_client(config)
        return isinstance(client, BatchChatModelClient)
    except Exception:
        return False


async def _run_extraction_batch_submit_phase(
    job_id: str, payload_json: dict[str, object]
) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="extraction_batch_submit")


async def _run_extraction_batch_collect_phase(
    job_id: str, payload_json: dict[str, object]
) -> None:
    payload = _payload_from_json(payload_json)
    session_factory = get_session_factory()
    async with session_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="extraction_batch_collect")


async def _poll_extraction_batch_once(
    job_id: str, payload_json: dict[str, object]
) -> bool:
    """Return True if the extraction batch is complete; False if still processing."""
    try:
        payload = _payload_from_json(payload_json)
        tc = payload.teacher_config
        if tc is None:
            return True
        phase_cfg = tc.for_phase("instruction_extraction")
        config = resolve_provider_config(
            phase_cfg.provider,
            phase_cfg.model,
            role="teacher",
            settings=get_settings(),
        )
        client = get_chat_client(config)
        if not isinstance(client, BatchChatModelClient):
            return True

        session_factory = get_session_factory()
        async with session_factory() as session:
            batch_job = await get_batch_job_by_stage(
                session, job_id, PipelineStage.EXTRACTING_RULES, strategy=None
            )
        if batch_job is None:
            return True

        poll_status = await client.poll_batch(batch_job.provider_batch_id, config)
        return not poll_status.processing
    except Exception:
        return True  # treat errors as "done" so the workflow advances to collect


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


@_step_decorator("submit_extraction_batch")
async def _submit_extraction_batch_step(
    job_id: str, payload_json: dict[str, object]
) -> None:
    await _run_extraction_batch_submit_phase(job_id, payload_json)


@_step_decorator_bool("poll_extraction_batch")
async def _poll_extraction_batch_step(
    job_id: str, payload_json: dict[str, object]
) -> bool:
    return await _poll_extraction_batch_once(job_id, payload_json)


@_step_decorator("collect_extraction_batch")
async def _collect_extraction_batch_step(
    job_id: str, payload_json: dict[str, object]
) -> None:
    await _run_extraction_batch_collect_phase(job_id, payload_json)


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

    if _extraction_batch_enabled(payload_json):
        # Submit, durably poll, then collect before continuing the compile chain.
        await _submit_extraction_batch_step(job_id, payload_json)

        poll_interval = get_settings().batch_poll_interval_seconds
        while True:
            if DBOS is not None:
                await _await_if_needed(DBOS.sleep(poll_interval))
            else:
                await asyncio.sleep(poll_interval)
            done = await _poll_extraction_batch_step(job_id, payload_json)
            if done:
                break

        await _collect_extraction_batch_step(job_id, payload_json)
    else:
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
