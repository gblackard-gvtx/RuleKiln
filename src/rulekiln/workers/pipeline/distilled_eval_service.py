"""DistilledEvaluationService: evaluate each distilled-prompt strategy against eval cases."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.eval_case_results import (
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.db.repositories.jobs import get_selected_synthesized_rules_for_job, is_job_cancelled
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.evaluator import evaluate_prompt
from rulekiln.pipeline.prompt_compiler import compile_prompt, count_tokens_approx
from rulekiln.pipeline.quality_gates import resolve_max_prompt_tokens
from rulekiln.schemas.pipeline import CaseEvalResult
from rulekiln.workers.pipeline.artifacts_state import PipelineArtifactsState
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.db_mappers import (
    _build_eval_case_upsert_payload,
    _db_synth_to_schema,
    _eval_case_record_to_schema,
)
from rulekiln.workers.pipeline.evaluation_persistence import _insert_eval_run_if_missing
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.seeding import _evaluation_bootstrap_seed
from rulekiln.workers.pipeline.stage_guard import StageGuard
from rulekiln.workers.pipeline.strategy_utils import DISTILLED_STRATEGIES

logger = get_logger(__name__)


class DistilledEvaluationService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        state: PipelineArtifactsState,
        *,
        phase: str,
    ) -> None:
        distilled_strategies: tuple[str, ...]
        if phase == "evaluate_dbscan":
            distilled_strategies = ("dbscan",)
        elif phase == "evaluate_hdbscan":
            distilled_strategies = ("hdbscan",)
        elif phase == "full":
            distilled_strategies = DISTILLED_STRATEGIES
        else:
            distilled_strategies = ()

        from rulekiln.providers.tracking import (  # noqa: PLC0415
            ModelCallContext,
            set_tracking_context,
        )

        student_profile = ctx.payload.student.provider_profile
        guard = StageGuard(self._session, ctx.job_id)

        for strategy in distilled_strategies:
            if await guard.is_complete(PipelineStage.EVALUATING_DISTILLED, strategy=strategy):
                continue

            db_synth = await get_selected_synthesized_rules_for_job(
                self._session, ctx.job_id, strategy
            )
            if not db_synth:
                logger.warning(
                    "distilled_strategy_skipped_no_rules",
                    job_id=ctx.job_id,
                    strategy=strategy,
                )
                state.strategy_metadata[strategy] = {
                    "unavailable": True,
                    "unavailable_reason": "no_selected_rules",
                }
                continue

            schemas = [_db_synth_to_schema(r) for r in db_synth]
            prompt_text, _ = compile_prompt(ctx.task, schemas, strategy)
            prompt_tokens = count_tokens_approx(prompt_text)
            max_prompt_tokens = resolve_max_prompt_tokens(
                ctx.task.quality_gates, ctx.settings.default_quality_gate
            )
            logger.info(
                "distilled_prompt_compiled",
                job_id=ctx.job_id,
                strategy=strategy,
                synthesized_rules_count=len(schemas),
                prompt_token_count=prompt_tokens,
                max_prompt_tokens=max_prompt_tokens,
                compiled_prompt_preview=prompt_text[:300],
            )
            if prompt_tokens > max_prompt_tokens:
                # An over-budget prompt would fail the token gate anyway; calling
                # the student per case just burns one provider error per case
                # (typically HTTP 400 context overflow → 100% malformed). Skip
                # evaluation and report the strategy as ineligible instead.
                logger.warning(
                    "distilled_strategy_skipped_prompt_over_budget",
                    job_id=ctx.job_id,
                    strategy=strategy,
                    prompt_token_count=prompt_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    synthesized_rules_count=len(schemas),
                )
                state.strategy_metadata[strategy] = {
                    "unavailable": True,
                    "unavailable_reason": (
                        f"prompt_tokens {prompt_tokens} exceeds "
                        f"max_prompt_tokens {max_prompt_tokens}; evaluation skipped"
                    ),
                    "prompt_token_count": prompt_tokens,
                    "synthesized_rules_count": len(schemas),
                }
                continue

            await guard.start(PipelineStage.EVALUATING_DISTILLED)
            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.EVALUATING_DISTILLED,
                    role="student",
                    provider_profile=student_profile,
                    provider=bundle.student_config.provider,
                    model=bundle.student_config.model,
                    strategy=strategy,
                ),
                ctx.collector,
            )
            existing_rows = await get_eval_case_results(
                self._session,
                job_id=ctx.job_id,
                student_id=bundle.eval_student_id,
                strategy=strategy,
                split=ctx.eval_split,
            )
            completed_case_results = {
                row.case_id: _eval_case_record_to_schema(row) for row in existing_rows
            }

            async def _persist_strategy_case(
                case_result: CaseEvalResult,
                *,
                strategy_name: str = strategy,
            ) -> None:
                case = ctx.case_by_id.get(case_result.case_id)
                if case is None:
                    return
                payload_row = _build_eval_case_upsert_payload(
                    job_id=ctx.job_id,
                    student_id=bundle.eval_student_id,
                    strategy=strategy_name,
                    split=ctx.eval_split,
                    case=case,
                    result=case_result,
                )
                await upsert_eval_case_result(self._session, payload_row)

            async def _check_cancelled() -> bool:
                return await is_job_cancelled(self._session, ctx.job_id)

            ev = await evaluate_prompt(
                prompt_text,
                ctx.eval_cases,
                ctx.task,
                bundle.student_chat,
                bundle.student_config,
                strategy=strategy,
                split=ctx.eval_split,
                completed_case_results=completed_case_results,
                on_case_result=_persist_strategy_case,
                bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                check_cancelled=_check_cancelled,
                bootstrap_seed=_evaluation_bootstrap_seed(
                    job_id=ctx.job_id,
                    strategy=strategy,
                    split=ctx.eval_split,
                    seed_offset=ctx.evaluation_bootstrap_seed_offset,
                ),
            )
            state.eval_map[strategy] = ev
            logger.info(
                "eval_complete",
                job_id=ctx.job_id,
                strategy=strategy,
                split=ctx.eval_split,
                malformed_output_rate=ev.malformed_output_rate,
                accuracy=ev.accuracy,
                macro_f1=ev.macro_f1,
                weighted_case_score=ev.weighted_case_score,
            )
            await _insert_eval_run_if_missing(
                self._session,
                job_id=ctx.job_id,
                split=ctx.eval_split,
                strategy=strategy,
                result=ev,
            )
            await guard.complete(PipelineStage.EVALUATING_DISTILLED, strategy=strategy)
