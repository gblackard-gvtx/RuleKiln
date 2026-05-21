"""Job and artifact repository helpers."""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import (
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
    result = await session.execute(
        select(DistillationJob).where(DistillationJob.id == job_id)
    )
    return result.scalar_one_or_none()


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


async def set_mlflow_run_id(
    session: AsyncSession, job_id: str, run_id: str
) -> None:
    await session.execute(
        update(DistillationJob)
        .where(DistillationJob.id == job_id)
        .values(mlflow_run_id=run_id)
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

async def bulk_insert_rule_clusters(
    session: AsyncSession, clusters: list[RuleCluster]
) -> None:
    session.add_all(clusters)
    await session.commit()


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

async def insert_prompt_version(
    session: AsyncSession, prompt_version: PromptVersion
) -> None:
    session.add(prompt_version)
    await session.commit()


async def get_selected_prompt_version(
    session: AsyncSession, job_id: str
) -> PromptVersion | None:
    result = await session.execute(
        select(PromptVersion).where(
            PromptVersion.job_id == job_id,
            PromptVersion.is_selected == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def mark_prompt_version_selected(
    session: AsyncSession, job_id: str, strategy: str
) -> None:
    await session.execute(
        update(PromptVersion)
        .where(PromptVersion.job_id == job_id)
        .values(is_selected=False)
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
