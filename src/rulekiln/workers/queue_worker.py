"""Postgres-backed queue worker: claims jobs and runs the distillation pipeline."""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid

import structlog

from rulekiln.config.settings import get_settings
from rulekiln.db.repositories.jobs import (
    claim_next_job,
    complete_job,
    fail_job,
    recover_expired_leases,
    renew_lease,
)
from rulekiln.db.session import get_session_factory
from rulekiln.schemas.job import DistillationRequest
from rulekiln.workers.distillation_worker import run_distillation_pipeline

logger = structlog.get_logger(__name__)

_SHUTDOWN = False


def _install_signal_handlers() -> None:
    def _handle(_sig: int, _frame: object) -> None:
        global _SHUTDOWN  # noqa: PLW0603
        logger.info("worker_shutdown_signal_received")
        _SHUTDOWN = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


async def _lease_renewer(
    job_id: str,
    worker_id: str,
    lease_seconds: int,
    stop_event: asyncio.Event,
) -> None:
    """Periodically renews the job lease until stop_event is set."""
    renew_interval = max(10, lease_seconds // 3)
    session_factory = get_session_factory()
    while not stop_event.is_set():
        await asyncio.sleep(renew_interval)
        if stop_event.is_set():
            break
        async with session_factory() as session:
            await renew_lease(session, job_id, worker_id, lease_seconds)
        logger.debug("lease_renewed", job_id=job_id, worker_id=worker_id)


async def worker_loop(worker_id: str) -> None:
    """Main loop: claim jobs, run pipeline, repeat until shutdown."""
    settings = get_settings()
    session_factory = get_session_factory()

    logger.info("worker_started", worker_id=worker_id)

    while not _SHUTDOWN:
        # Recover any jobs with expired leases before claiming
        async with session_factory() as session:
            retried, failed_count = await recover_expired_leases(session)
        if retried or failed_count:
            logger.info(
                "leases_recovered",
                worker_id=worker_id,
                retried=retried,
                failed=failed_count,
            )

        # Claim the next available job
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
        log.info("job_claimed")

        stop_renewer = asyncio.Event()
        renewer_task = asyncio.create_task(
            _lease_renewer(job.id, worker_id, settings.worker_lease_seconds, stop_renewer)
        )

        try:
            payload = DistillationRequest.model_validate(job.request_json)
            async with session_factory() as session:
                from rulekiln.workers.distillation_worker import _run  # noqa: PLC0415
                await _run(session, job.id, payload)
            async with session_factory() as session:
                await complete_job(session, job.id)
            log.info("job_completed")
        except Exception as exc:
            log.error("job_failed", error=str(exc))
            async with session_factory() as session:
                await fail_job(session, job.id, error_message=str(exc))
        finally:
            stop_renewer.set()
            renewer_task.cancel()
            try:
                await renewer_task
            except asyncio.CancelledError:
                pass


def main() -> None:
    """CLI entrypoint: rulekiln-worker."""
    _install_signal_handlers()
    worker_id = str(uuid.uuid4())
    asyncio.run(worker_loop(worker_id))
    sys.exit(0)


if __name__ == "__main__":
    main()
