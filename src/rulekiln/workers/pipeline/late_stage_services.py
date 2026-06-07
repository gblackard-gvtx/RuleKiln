"""Late-stage pipeline services: strategy selection through pruning optimization.

StrategySelectionService  – mark selected prompt version, log
FailureAnalysisService    – analyse failure patterns on the selected strategy
RefinementLoopService     – iterative rule refinement with teacher model
RuleAblationService       – ablate individual rules to measure contribution
PruningOptimizationService – two-pass pruning-budget optimizer
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.jobs import (
    get_selected_synthesized_rules_for_job,
    is_stage_complete,
    mark_prompt_version_selected,
    mark_stage_complete,
    update_job_status,
)
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.failure_analysis import FailureAnalysisResult, analyze_failures
from rulekiln.schemas.pipeline import (
    PruningModeComparison,
    RuleAblationArtifact,
    StrategyComparison,
)
from rulekiln.workers.pipeline.artifacts_state import PipelineArtifactsState
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.db_mappers import _db_synth_to_schema
from rulekiln.workers.pipeline.paths import _artifact_root
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.refinement_ablation import (
    _run_refinement_loop,
    _run_rule_ablation,
    _run_two_pass_optimizer,
)
from rulekiln.workers.pipeline.strategy_utils import DISTILLED_STRATEGIES

logger = get_logger(__name__)


class StrategySelectionService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        comparison: StrategyComparison,
        selected_strategy: str,
    ) -> None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.SELECTING_STRATEGY):
            return
        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.SELECTING_STRATEGY
        )
        if selected_strategy in DISTILLED_STRATEGIES:
            await mark_prompt_version_selected(self._session, ctx.job_id, selected_strategy)
        logger.info(
            "strategy_selected",
            job_id=ctx.job_id,
            strategy=selected_strategy,
            reason=comparison.selection_reason,
        )
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.SELECTING_STRATEGY)


class FailureAnalysisService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        state: PipelineArtifactsState,
        selected_strategy: str,
    ) -> FailureAnalysisResult | None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.ANALYZING_FAILURES):
            return None
        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.ANALYZING_FAILURES
        )
        failure_analysis_result: FailureAnalysisResult | None = None
        selected_eval = state.eval_map.get(selected_strategy)
        if selected_eval and selected_strategy in DISTILLED_STRATEGIES:
            db_synth = await get_selected_synthesized_rules_for_job(
                self._session, ctx.job_id, selected_strategy
            )
            selected_schemas = [_db_synth_to_schema(r) for r in db_synth]
            failure_analysis_result = analyze_failures(
                state.baseline_scaffold_eval or state.baseline_eval,
                selected_eval,
                selected_schemas,
                list(ctx.case_by_id.values()),
            )
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.ANALYZING_FAILURES)
        return failure_analysis_result


class RefinementLoopService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        state: PipelineArtifactsState,
        selected_strategy: str,
        failure_analysis_result: FailureAnalysisResult | None,
    ) -> None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.REFINING_RULES):
            return
        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.REFINING_RULES
        )
        _refinement_blocked_reason: str | None = None
        eval_split = ctx.eval_split
        if eval_split in ("test", "golden"):
            _refinement_blocked_reason = (
                f"eval_split={eval_split!r} — refinement must not"
                " optimize against test or golden data"
            )
        elif (
            eval_split == "train"
            and ctx.split_policy.fallback_warning
            and not ctx.task.allow_refinement_on_train_split
        ):
            _refinement_blocked_reason = (
                "eval_split='train' due to missing validation split and "
                "allow_refinement_on_train_split is not set — "
                "refinement would double-use train data already used for extraction"
            )
        if _refinement_blocked_reason is not None:
            logger.warning(
                "refinement_disabled_unsafe_split",
                job_id=ctx.job_id,
                eval_split=eval_split,
                reason=_refinement_blocked_reason,
            )
        _golden_cases_for_loop = [c for c in ctx.cases if c.split == "golden"]
        if (
            ctx.task.enable_refinement_loop
            and failure_analysis_result is not None
            and selected_strategy in DISTILLED_STRATEGIES
            and _refinement_blocked_reason is None
        ):
            selected_eval_for_loop = state.eval_map.get(selected_strategy)
            if selected_eval_for_loop is not None:
                _cr = ctx.settings.default_conflict_resolution
                _eff_max_iterations = (
                    ctx.task.refinement_max_iterations
                    if ctx.task.refinement_max_iterations is not None
                    else _cr.refinement_max_iterations
                )
                _eff_max_failure_cases = (
                    ctx.task.refinement_max_failure_cases
                    if ctx.task.refinement_max_failure_cases is not None
                    else _cr.max_failure_cases_per_iteration
                )
                _eff_max_success_cases = (
                    ctx.task.refinement_max_success_cases
                    if ctx.task.refinement_max_success_cases is not None
                    else _cr.max_success_cases_per_iteration
                )
                _eff_holdout_fraction = (
                    ctx.task.validation_holdout_fraction
                    if ctx.task.validation_holdout_fraction is not None
                    else _cr.validation_holdout_fraction
                )
                _task_gates = ctx.task.quality_gates
                _eff_min_metric_delta = float(
                    _task_gates.get(
                        "min_metric_delta",
                        ctx.settings.default_quality_gate.min_metric_delta,
                    )
                )
                _eff_max_golden_failures = int(
                    _task_gates.get(
                        "max_golden_failures",
                        ctx.settings.default_quality_gate.max_golden_failures,
                    )
                )
                _, refined_eval = await _run_refinement_loop(
                    session=self._session,
                    job_id=ctx.job_id,
                    task=ctx.task,
                    artifact_root_path=_artifact_root(ctx.settings.artifact_root, ctx.job_id),
                    failure_analysis_result=failure_analysis_result,
                    selected_strategy=selected_strategy,
                    current_eval=selected_eval_for_loop,
                    eval_cases=ctx.eval_cases,
                    eval_split=eval_split,
                    case_by_id=ctx.case_by_id,
                    golden_cases=_golden_cases_for_loop,
                    teacher_chat=bundle.refinement_teacher_chat,
                    teacher_config=bundle.refinement_teacher_config,
                    student_chat=bundle.student_chat,
                    student_config=bundle.student_config,
                    primary_metric=ctx.primary_metric,
                    bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                    bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                    bootstrap_seed=ctx.evaluation_bootstrap_seed_offset,
                    effective_max_iterations=_eff_max_iterations,
                    effective_max_failure_cases=_eff_max_failure_cases,
                    effective_max_success_cases=_eff_max_success_cases,
                    effective_holdout_fraction=_eff_holdout_fraction,
                    effective_min_metric_delta=_eff_min_metric_delta,
                    effective_max_golden_failures=_eff_max_golden_failures,
                )
                if refined_eval is not None:
                    state.eval_map[selected_strategy] = refined_eval
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.REFINING_RULES)


class RuleAblationService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        state: PipelineArtifactsState,
        selected_strategy: str,
    ) -> RuleAblationArtifact | None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.ABLATING_RULES):
            return None
        ablation_artifact = await _run_rule_ablation(
            self._session,
            job_id=ctx.job_id,
            task=ctx.task,
            selected_strategy=selected_strategy,
            eval_cases=ctx.eval_cases,
            eval_split=ctx.eval_split,
            case_by_id=ctx.case_by_id,
            student_id=bundle.eval_student_id,
            student_chat=bundle.student_chat,
            student_config=bundle.student_config,
            primary_metric=ctx.primary_metric,
            bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
            bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
            bootstrap_seed_offset=ctx.evaluation_bootstrap_seed_offset,
            eval_map=state.eval_map,
        )
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.ABLATING_RULES)
        return ablation_artifact


class PruningOptimizationService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        state: PipelineArtifactsState,
        selected_strategy: str,
        ablation_artifact: RuleAblationArtifact | None,
    ) -> PruningModeComparison | None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.OPTIMIZING_PRUNING):
            return None
        pruning_mode_comparison = await _run_two_pass_optimizer(
            self._session,
            job_id=ctx.job_id,
            task=ctx.task,
            selected_strategy=selected_strategy,
            eval_cases=ctx.eval_cases,
            eval_split=ctx.eval_split,
            case_by_id=ctx.case_by_id,
            student_id=bundle.eval_student_id,
            student_chat=bundle.student_chat,
            student_config=bundle.student_config,
            primary_metric=ctx.primary_metric,
            bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
            bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
            bootstrap_seed_offset=ctx.evaluation_bootstrap_seed_offset,
            eval_map=state.eval_map,
            ablation_artifact=ablation_artifact,
        )
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.OPTIMIZING_PRUNING)
        return pruning_mode_comparison
