"""Output retrieval routes: prompt, rules, eval-report for a completed job."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.jobs import (
    get_eval_runs_for_job,
    get_job,
    get_selected_prompt_version,
    get_synthesized_rules_for_job,
)
from rulekiln.db.session import get_db_session
from rulekiln.schemas.pipeline import EvalResult, SynthesizedRuleSchema

router = APIRouter(prefix="/jobs", tags=["outputs"])


class PromptResponse(BaseModel):
    job_id: str
    strategy: str
    version: str
    prompt_hash: str
    system_prompt: str


class RulesResponse(BaseModel):
    job_id: str
    strategy: str
    rules: list[SynthesizedRuleSchema]


class EvalReportResponse(BaseModel):
    job_id: str
    eval_runs: list[EvalResult]


@router.get("/{job_id}/prompt", response_model=PromptResponse)
async def get_job_prompt(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> PromptResponse:
    _assert_job_exists_and_done(await get_job(session, job_id), job_id)
    pv = await get_selected_prompt_version(session, job_id)
    if pv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No selected prompt version found for job {job_id!r}.",
        )
    return PromptResponse(
        job_id=job_id,
        strategy=pv.strategy,
        version=pv.version,
        prompt_hash=pv.prompt_hash,
        system_prompt=pv.system_prompt,
    )


@router.get("/{job_id}/rules", response_model=RulesResponse)
async def get_job_rules(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RulesResponse:
    _assert_job_exists_and_done(await get_job(session, job_id), job_id)
    pv = await get_selected_prompt_version(session, job_id)
    strategy = pv.strategy if pv is not None else "hdbscan"
    db_rules = await get_synthesized_rules_for_job(session, job_id, strategy)
    from rulekiln.schemas.pipeline import OutcomeCondition

    rules = [
        SynthesizedRuleSchema(
            topic=r.topic,
            applies_when=list(r.applies_when or []),
            outcome_conditions={
                k: OutcomeCondition.model_validate(v)
                for k, v in (r.outcome_conditions or {}).items()
            },
            tie_breakers=list(r.tie_breakers or []),
            priority=r.priority,
            source_case_ids=list(r.source_case_ids or []),
            source_micro_rule_ids=list(r.source_micro_rule_ids or []),
        )
        for r in db_rules
    ]
    return RulesResponse(job_id=job_id, strategy=strategy, rules=rules)


@router.get("/{job_id}/eval-report", response_model=EvalReportResponse)
async def get_job_eval_report(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> EvalReportResponse:
    _assert_job_exists_and_done(await get_job(session, job_id), job_id)
    db_runs = await get_eval_runs_for_job(session, job_id)
    eval_runs = [
        EvalResult(
            strategy=r.strategy,
            model=r.model,
            split=r.split,
            accuracy=r.accuracy,
            macro_f1=r.macro_f1,
            weighted_case_score=r.weighted_case_score,
            malformed_output_rate=r.malformed_output_rate,
            per_outcome_precision=dict(r.per_outcome_precision or {}),
            per_outcome_recall=dict(r.per_outcome_recall or {}),
            confusion_matrix={k: dict(v) for k, v in (r.confusion_matrix or {}).items()},
        )
        for r in db_runs
    ]
    return EvalReportResponse(job_id=job_id, eval_runs=eval_runs)


def _assert_job_exists_and_done(job: object, job_id: str) -> None:
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )
