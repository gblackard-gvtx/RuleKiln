"""DBOS backend worker entrypoint for the full distillation pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
import uuid

import structlog

from rulekiln.config.settings import get_settings
from rulekiln.db.repositories.jobs import (
    apply_job_failure_policy,
    claim_next_job,
    complete_job,
    recover_expired_leases,
    renew_lease,
)
from rulekiln.db.session import get_session_factory
from rulekiln.schemas.job import DistillationRequest
from rulekiln.workers.error_classification import classify_worker_error
from rulekiln.workers.dbos_runtime import ensure_dbos_runtime_launched

logger = structlog.get_logger(__name__)

_SHUTDOWN = False


def _install_signal_handlers() -> None:
    def _handle(_sig: int, _frame: object) -> None:
        global _SHUTDOWN  # noqa: PLW0603
        logger.info("dbos_worker_shutdown_signal_received")
        _SHUTDOWN = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


async def _lease_renewer(
    job_id: str,
    worker_id: str,
    lease_seconds: int,
    stop_event: asyncio.Event,
) -> None:
    renew_interval = max(10, lease_seconds // 3)
    session_factory = get_session_factory()
    while not stop_event.is_set():
        await asyncio.sleep(renew_interval)
        if stop_event.is_set():
            break
        async with session_factory() as session:
            await renew_lease(session, job_id, worker_id, lease_seconds)
        logger.debug("dbos_worker_lease_renewed", job_id=job_id, worker_id=worker_id)


async def worker_loop(worker_id: str) -> None:
    settings = get_settings()
    ensure_dbos_runtime_launched(settings)

    if settings.execution_backend != "dbos":
        logger.warning(
            "dbos_worker_running_with_non_dbos_backend",
            configured_backend=settings.execution_backend,
        )

    session_factory = get_session_factory()
    logger.info("dbos_worker_started", worker_id=worker_id)

    while not _SHUTDOWN:
        async with session_factory() as session:
            retried, failed_count = await recover_expired_leases(session)
        if retried or failed_count:
            logger.info(
                "dbos_worker_leases_recovered",
                worker_id=worker_id,
                retried=retried,
                failed=failed_count,
            )

        async with session_factory() as session:
            job = await claim_next_job(
                session,
                worker_id=worker_id,
                lease_seconds=settings.worker_lease_seconds,
            )

        if job is None:
            await asyncio.sleep(settings.worker_poll_interval_seconds)
            continue

        log = logger.bind(job_id=job.id, worker_id=worker_id)
        log.info("dbos_worker_job_claimed")

        stop_renewer = asyncio.Event()
        renewer_task = asyncio.create_task(
            _lease_renewer(job.id, worker_id, settings.worker_lease_seconds, stop_renewer)
        )

        try:
            payload = DistillationRequest.model_validate(job.request_json)
            from rulekiln.workers.dbos_workflow import run_dbos_stage_workflow  # noqa: PLC0415

            await run_dbos_stage_workflow(job.id, payload)
            async with session_factory() as session:
                await complete_job(session, job.id)
            log.info("dbos_worker_job_completed")
        except Exception as exc:
            classification = classify_worker_error(exc)
            log.error(
                "dbos_worker_job_failed",
                error=str(exc),
                error_type=classification.error_type,
                retryable=classification.retryable,
            )
            async with session_factory() as session:
                status = await apply_job_failure_policy(
                    session,
                    job.id,
                    error_message=str(exc),
                    retryable=classification.retryable,
                    retry_backoff_seconds=settings.worker_retry_backoff_seconds,
                )
            log.info("dbos_worker_failure_policy_applied", resulting_status=status)
        finally:
            stop_renewer.set()
            renewer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewer_task


def main() -> None:
    _install_signal_handlers()
    worker_id = str(uuid.uuid4())
    asyncio.run(worker_loop(worker_id))
    sys.exit(0)


if __name__ == "__main__":
    main()
