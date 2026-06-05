"""Job and artifact repository helpers."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import (
    BatchJob,
    Case,
    DistillationJob,
    EvalRun,
    MicroRule,
    PromptVersion,
    RuleCluster,
    StageMarker,
    SynthesizedRule,
)

# ── Job ──────────────────────────────────────────────────────


async def create_job(session: AsyncSession, job: DistillationJob) -> None:
    session.add(job)
    await session.commit()


async def get_job(session: AsyncSession, job_id: str) -> DistillationJob | None:
    result = await session.execute(select(DistillationJob).where(DistillationJob.id == job_id))
    return result.scalar_one_or_none()


async def list_recent_jobs(
    session: AsyncSession,
    limit: int = 50,
    include_drafts: bool = False,
) -> list[DistillationJob]:
    """Return the most recent jobs ordered by created_at descending."""
    query = select(DistillationJob)
    if not include_drafts:
        query = query.where(DistillationJob.status != "draft")
    result = await session.execute(query.order_by(DistillationJob.created_at.desc()).limit(limit))
    return list(result.scalars().all())


async def update_job_status(
    session: AsyncSession,
    job_id: str,
    status: str,
    stage: str | None = None,
    error_message: str | None = None,
) -> None:
    values: dict[str, str | None] = {"status": status}
    if stage is not None:
        values["stage"] = stage
    if error_message is not None:
        values["error_message"] = error_message
    await session.execute(
        update(DistillationJob).where(DistillationJob.id == job_id).values(**values)
    )
    await session.commit()


async def set_mlflow_run_id(session: AsyncSession, job_id: str, run_id: str) -> None:
    await session.execute(
        update(DistillationJob).where(DistillationJob.id == job_id).values(mlflow_run_id=run_id)
    )
    await session.commit()


# ── Cases ─────────────────────────────────────────────────────


async def bulk_insert_cases(session: AsyncSession, cases: list[Case]) -> None:
    session.add_all(cases)
    await session.commit()


async def get_cases_for_job(session: AsyncSession, job_id: str) -> list[Case]:
    result = await session.execute(select(Case).where(Case.job_id == job_id))
    return list(result.scalars().all())


# ── Stage markers ─────────────────────────────────────────────


async def mark_stage_complete(
    session: AsyncSession,
    job_id: str,
    stage: str,
    strategy: str | None = None,
    artifact_type: str | None = None,
) -> None:
    existing = await session.execute(
        select(StageMarker).where(
            StageMarker.job_id == job_id,
            StageMarker.stage == stage,
            StageMarker.strategy == strategy,
            StageMarker.artifact_type == artifact_type,
        )
    )
    if existing.scalar_one_or_none() is None:
        session.add(
            StageMarker(
                job_id=job_id,
                stage=stage,
                strategy=strategy,
                artifact_type=artifact_type,
            )
        )
        await session.commit()


async def is_stage_complete(
    session: AsyncSession,
    job_id: str,
    stage: str,
    strategy: str | None = None,
    artifact_type: str | None = None,
) -> bool:
    result = await session.execute(
        select(StageMarker).where(
            StageMarker.job_id == job_id,
            StageMarker.stage == stage,
            StageMarker.strategy == strategy,
            StageMarker.artifact_type == artifact_type,
        )
    )
    return result.scalar_one_or_none() is not None


# ── Micro rules ───────────────────────────────────────────────


async def bulk_insert_micro_rules(session: AsyncSession, rules: list[MicroRule]) -> None:
    session.add_all(rules)
    await session.commit()


async def get_micro_rules_for_job(session: AsyncSession, job_id: str) -> list[MicroRule]:
    result = await session.execute(select(MicroRule).where(MicroRule.job_id == job_id))
    return list(result.scalars().all())


# ── Rule clusters ─────────────────────────────────────────────


async def bulk_insert_rule_clusters(session: AsyncSession, clusters: list[RuleCluster]) -> None:
    session.add_all(clusters)
    await session.commit()


async def get_rule_clusters_for_job(
    session: AsyncSession, job_id: str, strategy: str
) -> list[RuleCluster]:
    """Return all rule clusters for a job+strategy (for provenance cluster_id lookup)."""
    result = await session.execute(
        select(RuleCluster).where(
            RuleCluster.job_id == job_id,
            RuleCluster.strategy == strategy,
        )
    )
    return list(result.scalars().all())


# ── Synthesized rules ─────────────────────────────────────────


async def bulk_insert_synthesized_rules(
    session: AsyncSession, rules: list[SynthesizedRule]
) -> None:
    session.add_all(rules)
    await session.commit()


async def get_synthesized_rules_for_job(
    session: AsyncSession, job_id: str, strategy: str
) -> list[SynthesizedRule]:
    result = await session.execute(
        select(SynthesizedRule).where(
            SynthesizedRule.job_id == job_id,
            SynthesizedRule.strategy == strategy,
        )
    )
    return list(result.scalars().all())


# ── Prompt versions ───────────────────────────────────────────


async def insert_prompt_version(session: AsyncSession, prompt_version: PromptVersion) -> None:
    session.add(prompt_version)
    await session.commit()


async def get_selected_prompt_version(session: AsyncSession, job_id: str) -> PromptVersion | None:
    result = await session.execute(
        select(PromptVersion).where(
            PromptVersion.job_id == job_id,
            PromptVersion.is_selected == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def mark_prompt_version_selected(session: AsyncSession, job_id: str, strategy: str) -> None:
    await session.execute(
        update(PromptVersion).where(PromptVersion.job_id == job_id).values(is_selected=False)
    )
    await session.execute(
        update(PromptVersion)
        .where(PromptVersion.job_id == job_id, PromptVersion.strategy == strategy)
        .values(is_selected=True)
    )
    await session.commit()


# ── Eval runs ─────────────────────────────────────────────────


async def insert_eval_run(session: AsyncSession, eval_run: EvalRun) -> None:
    session.add(eval_run)
    await session.commit()


async def get_eval_runs_for_job(session: AsyncSession, job_id: str) -> list[EvalRun]:
    result = await session.execute(select(EvalRun).where(EvalRun.job_id == job_id))
    return list(result.scalars().all())


# ── Synthesized rule updates ──────────────────────────────────


async def update_synthesized_rule_conflict(
    session: AsyncSession,
    rule_id: str,
    has_conflicts: bool,
    conflict_summary: str | None,
    conflicting_micro_rule_ids: list[str],
) -> None:
    await session.execute(
        update(SynthesizedRule)
        .where(SynthesizedRule.id == rule_id)
        .values(
            has_conflicts=has_conflicts,
            conflict_summary=conflict_summary,
            conflicting_micro_rule_ids=conflicting_micro_rule_ids,
        )
    )
    await session.commit()


async def update_synthesized_rule_pruning(
    session: AsyncSession,
    rule_id: str,
    is_pruned: bool,
    pruning_reason: str | None,
    support_count: int,
    support_ratio: float,
    golden_case_backed: bool,
    estimated_token_count: int,
) -> None:
    await session.execute(
        update(SynthesizedRule)
        .where(SynthesizedRule.id == rule_id)
        .values(
            is_pruned=is_pruned,
            pruning_reason=pruning_reason,
            support_count=support_count,
            support_ratio=support_ratio,
            golden_case_backed=golden_case_backed,
            estimated_token_count=estimated_token_count,
        )
    )
    await session.commit()


async def get_selected_synthesized_rules_for_job(
    session: AsyncSession, job_id: str, strategy: str
) -> list[SynthesizedRule]:
    """Return only non-pruned synthesized rules for a given strategy."""
    result = await session.execute(
        select(SynthesizedRule).where(
            SynthesizedRule.job_id == job_id,
            SynthesizedRule.strategy == strategy,
            SynthesizedRule.is_pruned == False,  # noqa: E712
        )
    )
    return list(result.scalars().all())


# ── Postgres queue operations ─────────────────────────────────


async def claim_next_job(
    session: AsyncSession,
    worker_id: str,
    lease_seconds: int,
) -> DistillationJob | None:
    """Claim the next pending job using FOR UPDATE SKIP LOCKED.

    Returns the claimed job or None if the queue is empty.
    """
    result = await session.execute(
        text("""
            WITH next_job AS (
                SELECT id
                FROM distillation_jobs
                WHERE queue_status = 'pending'
                  AND next_run_at <= now()
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE distillation_jobs j
            SET queue_status    = 'running',
                status          = 'running',
                locked_by       = :worker_id,
                locked_at       = now(),
                lease_expires_at = now() + make_interval(secs => :lease_seconds),
                attempt_count   = attempt_count + 1,
                updated_at      = now()
            FROM next_job
            WHERE j.id = next_job.id
            RETURNING j.id
        """),
        {"worker_id": worker_id, "lease_seconds": lease_seconds},
    )
    row = result.fetchone()
    await session.commit()
    if row is None:
        return None
    return await get_job(session, str(row[0]))


async def renew_lease(
    session: AsyncSession,
    job_id: str,
    worker_id: str,
    lease_seconds: int,
) -> None:
    """Extend the lease on a running job owned by worker_id."""
    await session.execute(
        update(DistillationJob)
        .where(
            DistillationJob.id == job_id,
            DistillationJob.locked_by == worker_id,
            DistillationJob.queue_status == "running",
        )
        .values(
            lease_expires_at=datetime.now(tz=UTC) + timedelta(seconds=lease_seconds),
        )
    )
    await session.commit()


async def complete_job(session: AsyncSession, job_id: str) -> None:
    """Mark a job as completed in the queue."""
    await session.execute(
        update(DistillationJob)
        .where(DistillationJob.id == job_id)
        .values(
            queue_status="completed",
            status="completed",
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
        )
    )
    await session.commit()


async def cancel_job(
    session: AsyncSession,
    job_id: str,
    *,
    error_message: str = "Cancelled by operator.",
) -> None:
    """Mark a job as cancelled with terminal status and cleared queue lease state."""
    await session.execute(
        update(DistillationJob)
        .where(DistillationJob.id == job_id)
        .values(
            queue_status="failed",
            status="failed_terminal",
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            error_message=error_message,
        )
    )
    await session.commit()


async def fail_job(session: AsyncSession, job_id: str, error_message: str) -> None:
    """Mark a job as permanently failed in the queue."""
    await session.execute(
        update(DistillationJob)
        .where(DistillationJob.id == job_id)
        .values(
            queue_status="failed",
            status="failed",
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            error_message=error_message,
        )
    )
    await session.commit()


async def retry_job(
    session: AsyncSession,
    job_id: str,
    *,
    queue_backed: bool,
) -> str:
    """Requeue an existing job for manual retry.

    Returns the status written to distillation_jobs.status.
    """
    status = "waiting_for_retry" if queue_backed else "created"
    queue_status = "pending" if queue_backed else "created"
    await session.execute(
        update(DistillationJob)
        .where(DistillationJob.id == job_id)
        .values(
            status=status,
            queue_status=queue_status,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            next_run_at=datetime.now(tz=UTC),
            error_message=None,
        )
    )
    await session.commit()
    return status


async def apply_job_failure_policy(
    session: AsyncSession,
    job_id: str,
    *,
    error_message: str,
    retryable: bool,
    retry_backoff_seconds: int,
) -> str:
    """Apply retry classification to a failed running job.

    Returns the final status written to distillation_jobs.status.
    """
    job = await get_job(session, job_id)
    if job is None:
        return "failed_terminal"

    retry_budget_remaining = job.attempt_count < job.max_attempts
    if retryable and retry_budget_remaining:
        next_run_at = datetime.now(tz=UTC) + timedelta(seconds=retry_backoff_seconds)
        await session.execute(
            update(DistillationJob)
            .where(DistillationJob.id == job_id)
            .values(
                queue_status="pending",
                status="waiting_for_retry",
                next_run_at=next_run_at,
                locked_by=None,
                locked_at=None,
                lease_expires_at=None,
                error_message=error_message,
            )
        )
        await session.commit()
        return "waiting_for_retry"

    final_status = "failed_retryable" if retryable else "failed_terminal"
    await session.execute(
        update(DistillationJob)
        .where(DistillationJob.id == job_id)
        .values(
            queue_status="failed",
            status=final_status,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            error_message=error_message,
        )
    )
    await session.commit()
    return final_status


async def recover_expired_leases(
    session: AsyncSession,
) -> tuple[int, int]:
    """Reset expired-lease jobs back to pending or fail them if max_attempts exceeded.

    Returns (retried_count, failed_count).
    """
    now = datetime.now(tz=UTC)

    # Return to pending if under max_attempts
    retried = await session.execute(
        update(DistillationJob)
        .where(
            DistillationJob.queue_status == "running",
            DistillationJob.lease_expires_at < now,
            DistillationJob.attempt_count < DistillationJob.max_attempts,
        )
        .values(
            queue_status="pending",
            status="waiting_for_retry",
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
        )
    )

    # Fail permanently if max_attempts exhausted
    failed = await session.execute(
        update(DistillationJob)
        .where(
            DistillationJob.queue_status == "running",
            DistillationJob.lease_expires_at < now,
            DistillationJob.attempt_count >= DistillationJob.max_attempts,
        )
        .values(
            queue_status="failed",
            status="failed_retryable",
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            error_message="Job exceeded maximum retry attempts.",
        )
    )

    await session.commit()
    retried_count = int(getattr(retried, "rowcount", 0) or 0)
    failed_count = int(getattr(failed, "rowcount", 0) or 0)
    return (retried_count, failed_count)


# ── Batch job CRUD ────────────────────────────────────────────────────────────


async def insert_batch_job(session: AsyncSession, batch_job: BatchJob) -> None:
    """Persist a new BatchJob record and flush to obtain the DB-assigned updated_at."""
    session.add(batch_job)
    await session.flush()


async def get_batch_job_by_stage(
    session: AsyncSession,
    job_id: str,
    stage: str,
    strategy: str | None,
) -> BatchJob | None:
    """Return the most-recently submitted BatchJob for a given stage/strategy, or None."""
    result = await session.execute(
        select(BatchJob)
        .where(
            BatchJob.job_id == job_id,
            BatchJob.stage == stage,
            BatchJob.strategy == strategy,
        )
        .order_by(BatchJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def update_batch_job(
    session: AsyncSession,
    batch_job_id: str,
    *,
    status: str | None = None,
    succeeded_count: int | None = None,
    errored_count: int | None = None,
    output_file_id: str | None = None,
    error_file_id: str | None = None,
    completed_at: datetime | None = None,
) -> None:
    """Patch mutable fields on an existing BatchJob."""
    values: dict[str, object] = {}
    if status is not None:
        values["status"] = status
    if succeeded_count is not None:
        values["succeeded_count"] = succeeded_count
    if errored_count is not None:
        values["errored_count"] = errored_count
    if output_file_id is not None:
        values["output_file_id"] = output_file_id
    if error_file_id is not None:
        values["error_file_id"] = error_file_id
    if completed_at is not None:
        values["completed_at"] = completed_at
    if not values:
        return
    await session.execute(update(BatchJob).where(BatchJob.id == batch_job_id).values(**values))
    await session.flush()
