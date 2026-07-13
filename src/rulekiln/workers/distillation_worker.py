"""Full distillation pipeline worker with stage orchestration, resume semantics, and idempotency."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.config.settings import get_settings
from rulekiln.db.models import (
    MicroRule,
)
from rulekiln.db.repositories.jobs import (
    get_selected_synthesized_rules_for_job,
    is_stage_complete,
    mark_stage_complete,
    update_job_status,
)
from rulekiln.db.session import get_session_factory
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.baseline_strategies import (
    BASELINE_FEW_SHOT_STRATEGY_TO_K,
    BASELINE_SCAFFOLD_STRATEGY,
    EMBEDDING_CENTROID_STRATEGY,
    EMBEDDING_KNN_STRATEGY_TO_K,
    RETRIEVAL_FEW_SHOT_STRATEGY,
)
from rulekiln.pipeline.evaluator import (
    evaluate_prompt,
)
from rulekiln.pipeline.prompt_compiler import (
    compile_prompt,
    count_tokens_approx,
)
from rulekiln.pipeline.quality_gates import check_quality_gates, resolve_max_prompt_tokens
from rulekiln.pipeline.statistics import (
    PairedComparisonArtifacts,
)
from rulekiln.pipeline.strategy_selection import build_strategy_comparison
from rulekiln.providers.tracking import (
    ModelCallContext,
    set_tracking_context,
)
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import (
    EvalResult,
    RuleAblationArtifact,
)
from rulekiln.workers.error_classification import format_worker_error_message
from rulekiln.workers.pipeline.artifacts_state import PipelineArtifactsState
from rulekiln.workers.pipeline.baseline_eval_service import BaselineEvaluationService
from rulekiln.workers.pipeline.classification_metrics import (
    _build_runtime_paired_comparison,
    _is_classification_task_mode,
    _refresh_eval_classification_statistics,
)
from rulekiln.workers.pipeline.compile_phase_stages import (
    PromptCompilationService,
    PruningStageService,
    StaticConflictReviewService,
)
from rulekiln.workers.pipeline.context_factory import PipelineContextFactory
from rulekiln.workers.pipeline.db_mappers import (
    _db_synth_to_schema,
)
from rulekiln.workers.pipeline.distilled_eval_service import DistilledEvaluationService
from rulekiln.workers.pipeline.early_stages import (
    ClusteringStageService,
    ExtractionStageService,
    RuleEmbeddingStageService,
    SynthesisStageService,
    ValidationStageService,
)
from rulekiln.workers.pipeline.evaluation_persistence import (
    _load_eval_result_from_db,
    _persist_collector_records,
)
from rulekiln.workers.pipeline.late_stage_services import (
    FailureAnalysisService,
    PruningOptimizationService,
    RefinementLoopService,
    RuleAblationService,
    StrategySelectionService,
)
from rulekiln.workers.pipeline.orchestrator import DistillationPipelineRunner
from rulekiln.workers.pipeline.output_stage_services import (
    ArtifactExportService,
    ClassroomEvaluationService,
    MlflowLoggingService,
)
from rulekiln.workers.pipeline.phase_plan import PipelinePhase
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle_factory import ProviderBundleFactory
from rulekiln.workers.pipeline.seeding import _evaluation_bootstrap_seed
from rulekiln.workers.pipeline.strategy_utils import (
    DISTILLED_STRATEGIES,
    _best_strategy_for_family,
    _strategy_family,
)

# Re-export PipelineStage so existing callers (tests, DBOS, etc.) keep working.
__all__ = ["PipelineStage"]

logger = get_logger(__name__)

BASELINE_STRATEGY = "baseline"
BASELINE_FEW_SHOT_STRATEGIES: tuple[str, ...] = tuple(BASELINE_FEW_SHOT_STRATEGY_TO_K.keys())
EMBEDDING_KNN_STRATEGIES: tuple[str, ...] = tuple(EMBEDDING_KNN_STRATEGY_TO_K.keys())
EMBEDDING_BASELINE_STRATEGIES: tuple[str, ...] = (
    EMBEDDING_CENTROID_STRATEGY,
    *EMBEDDING_KNN_STRATEGIES,
)
BASELINE_VARIANT_STRATEGIES: tuple[str, ...] = (
    BASELINE_SCAFFOLD_STRATEGY,
    *BASELINE_FEW_SHOT_STRATEGIES,
    *EMBEDDING_BASELINE_STRATEGIES,
    RETRIEVAL_FEW_SHOT_STRATEGY,
)


async def run_distillation_pipeline(
    job_id: str,
    payload: DistillationRequest,
) -> None:
    """Top-level background task driving the full pipeline stage-by-stage."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            await DistillationPipelineRunner(session).run(job_id, payload)
        except Exception as exc:
            error_message = format_worker_error_message(exc)
            logger.error("pipeline_failed", job_id=job_id, error=error_message)
            await update_job_status(
                session,
                job_id,
                status="failed",
                stage=PipelineStage.FAILED,
                error_message=error_message,
            )
            raise


async def run_pipeline_phase(
    session: AsyncSession,
    job_id: str,
    payload: DistillationRequest,
    *,
    phase: PipelinePhase,
) -> None:
    """Run one pipeline phase while preserving existing stage-marker idempotency guards."""
    await DistillationPipelineRunner(session).run(job_id, payload, phase=phase)


async def _run(
    session: AsyncSession,
    job_id: str,
    payload: DistillationRequest,
    *,
    phase: PipelinePhase = "full",
) -> None:  # noqa: C901
    settings = get_settings()
    ctx = PipelineContextFactory.create(job_id, payload, settings)

    # Local aliases consumed by the still-inline stages below.
    task = ctx.task
    cases = ctx.cases
    extraction_cases = ctx.extraction_cases
    eval_cases = ctx.eval_cases
    eval_split = ctx.eval_split
    collector = ctx.collector
    evaluation_bootstrap_enabled = ctx.evaluation_bootstrap_enabled
    evaluation_bootstrap_iterations = ctx.evaluation_bootstrap_iterations
    evaluation_bootstrap_seed_offset = ctx.evaluation_bootstrap_seed_offset

    logger.info(
        "pipeline_started",
        job_id=job_id,
        phase=phase,
        task_mode=task.task_mode,
        n_cases=len(cases),
        n_eval_cases=len(eval_cases),
        n_extraction_cases=len(extraction_cases),
        student_profile=payload.student.provider_profile,
    )
    logger.info(
        "split_policy_resolved",
        job_id=job_id,
        split_counts=ctx.split_policy.split_counts,
        extraction_split=ctx.split_policy.extraction_split,
        extraction_case_count=len(extraction_cases),
        evaluation_split=eval_split,
        evaluation_case_count=len(eval_cases),
    )
    if ctx.split_policy.fallback_warning is not None:
        logger.warning(
            "evaluation_split_fallback",
            job_id=job_id,
            warning=ctx.split_policy.fallback_warning,
            split_counts=ctx.split_policy.split_counts,
        )

    run_validate = phase in {"full", "validate_project", "compile_prompts"}
    run_compile_phase = phase in {
        "full",
        "compile_prompts",
        "extraction_batch_submit",
        "extraction_batch_collect",
    }
    run_post_extraction = phase in {
        "full",
        "compile_prompts",
        "extraction_batch_collect",
    }
    run_aggregate = phase in {"full", "aggregate_evaluation_report"}

    student_profile = payload.student.provider_profile

    # ── Stage: validating_project ──────────────────────────────────────────
    if run_validate:
        await ValidationStageService(session).run(ctx)

    if phase == "validate_project":
        return

    bundle = ProviderBundleFactory.create(payload, settings)

    # Local aliases consumed by the still-inline stages below.
    student_config = bundle.student_config
    student_chat = bundle.student_chat
    eval_student_id = bundle.eval_student_id
    primary_metric = ctx.primary_metric

    if run_compile_phase:
        # ── Stages: extracting → synthesizing ────────────────────────────
        batch_submitted = await ExtractionStageService(session).run(ctx, bundle, phase=phase)
        if batch_submitted:
            return

        if not run_post_extraction:
            return

        _emb = await RuleEmbeddingStageService(session).run(ctx, bundle)
        _clust = await ClusteringStageService(session).run(ctx, _emb)
        rule_map: dict[str, MicroRule] = {r.id: r for r in _emb.db_micro_rules}
        await SynthesisStageService(session).run(ctx, bundle, _clust, rule_map)

        # ── Stages: reviewing_rule_conflicts → pruning → compiling_prompts ──
        await StaticConflictReviewService(session).run(ctx, bundle, rule_map)
        await PruningStageService(session).run(ctx)
        await PromptCompilationService(session).run(ctx)

    if phase == "compile_prompts":
        await _persist_collector_records(session, job_id, collector)
        return

    # ── Stage: evaluating_baseline ────────────────────────────────────────
    state = PipelineArtifactsState()
    await BaselineEvaluationService(session).run(ctx, bundle, state, phase=phase)

    baseline_prompt_tokens = state.baseline_prompt_tokens
    baseline_variant_prompts = state.baseline_variant_prompts
    baseline_variant_evals = state.baseline_variant_evals
    strategy_prompt_tokens = state.strategy_prompt_tokens
    strategy_metadata = state.strategy_metadata
    baseline_eval: EvalResult | None = state.baseline_eval
    baseline_scaffold_eval: EvalResult | None = state.baseline_scaffold_eval

    if phase == "evaluate_baseline":
        await _persist_collector_records(session, job_id, collector)
        return

    # ── Stage: evaluating_distilled ───────────────────────────────────────
    state.eval_map.update(baseline_variant_evals)
    eval_map = state.eval_map
    await DistilledEvaluationService(session).run(ctx, bundle, state, phase=phase)

    if phase in {"evaluate_dbscan", "evaluate_hdbscan"}:
        await _persist_collector_records(session, job_id, collector)
        return

    if phase == "extraction_batch_collect":
        await _persist_collector_records(session, job_id, collector)
        return

    if run_aggregate:
        paired_comparison_artifacts: PairedComparisonArtifacts | None = None
        ablation_artifact: RuleAblationArtifact | None = None

        if baseline_eval is None:
            baseline_eval = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=BASELINE_STRATEGY,
                split=eval_split,
            )
            state.baseline_eval = baseline_eval

        if baseline_scaffold_eval is None:
            baseline_scaffold_eval = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=BASELINE_SCAFFOLD_STRATEGY,
                split=eval_split,
            )
            if baseline_scaffold_eval is None and baseline_eval is not None:
                baseline_scaffold_eval = baseline_eval.model_copy(
                    update={
                        "strategy": BASELINE_SCAFFOLD_STRATEGY,
                        "prompt_token_count": baseline_prompt_tokens,
                    }
                )
            state.baseline_scaffold_eval = baseline_scaffold_eval

        if baseline_scaffold_eval is not None:
            eval_map.setdefault(BASELINE_SCAFFOLD_STRATEGY, baseline_scaffold_eval)

        for strategy in (*BASELINE_FEW_SHOT_STRATEGIES, *EMBEDDING_BASELINE_STRATEGIES):
            if strategy in eval_map:
                continue
            loaded = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=strategy,
                split=eval_split,
            )
            if loaded is not None:
                eval_map[strategy] = loaded

        if _is_classification_task_mode(task.task_mode):
            refreshed_eval_map: dict[str, EvalResult] = {}
            for strategy_name, strategy_eval in eval_map.items():
                refreshed_eval_map[strategy_name] = _refresh_eval_classification_statistics(
                    eval_result=strategy_eval,
                    eval_cases=eval_cases,
                    bootstrap_enabled=evaluation_bootstrap_enabled,
                    bootstrap_iterations=evaluation_bootstrap_iterations,
                    bootstrap_seed=_evaluation_bootstrap_seed(
                        job_id=job_id,
                        strategy=strategy_name,
                        split=eval_split,
                        seed_offset=evaluation_bootstrap_seed_offset,
                    ),
                )
            state.eval_map = refreshed_eval_map
            eval_map = state.eval_map
            if baseline_eval is not None and BASELINE_STRATEGY in eval_map:
                baseline_eval = eval_map[BASELINE_STRATEGY]
                state.baseline_eval = baseline_eval
            if baseline_scaffold_eval is not None and BASELINE_SCAFFOLD_STRATEGY in eval_map:
                baseline_scaffold_eval = eval_map[BASELINE_SCAFFOLD_STRATEGY]
                state.baseline_scaffold_eval = baseline_scaffold_eval

        if RETRIEVAL_FEW_SHOT_STRATEGY not in eval_map:
            loaded_retrieval = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                split=eval_split,
            )
            if loaded_retrieval is not None:
                eval_map[RETRIEVAL_FEW_SHOT_STRATEGY] = loaded_retrieval

        for strategy in DISTILLED_STRATEGIES:
            if strategy in eval_map:
                continue
            # Don't surface stale eval rows for strategies that had no rules —
            # they would pollute strategy comparison with bogus malformed results.
            db_synth = await get_selected_synthesized_rules_for_job(session, job_id, strategy)
            if not db_synth:
                continue
            loaded = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=strategy,
                split=eval_split,
            )
            if loaded is not None:
                eval_map[strategy] = loaded
                continue
            # No eval row despite having selected rules: the evaluate_* step
            # (a separate DBOS step with its own in-memory state) most likely
            # skipped this strategy for exceeding max_prompt_tokens. Recompute
            # the same check here so the report surfaces it as unavailable
            # instead of silently omitting it.
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            prompt_text, _ = compile_prompt(task, schemas, strategy)
            prompt_tokens = count_tokens_approx(prompt_text)
            max_prompt_tokens = resolve_max_prompt_tokens(
                task.quality_gates, settings.default_quality_gate
            )
            if prompt_tokens > max_prompt_tokens:
                strategy_metadata[strategy] = {
                    "unavailable": True,
                    "unavailable_reason": (
                        f"prompt_tokens {prompt_tokens} exceeds "
                        f"max_prompt_tokens {max_prompt_tokens}; evaluation skipped"
                    ),
                    "prompt_token_count": prompt_tokens,
                    "synthesized_rules_count": len(schemas),
                }

        # ── Stage: checking_quality_gates ─────────────────────────────────
        gate_map = state.gate_map
        token_counts = state.token_counts
        token_counts.update(strategy_prompt_tokens)
        token_counts.setdefault(BASELINE_SCAFFOLD_STRATEGY, baseline_prompt_tokens)
        compiled_prompts = state.compiled_prompts
        compiled_prompts.update(baseline_variant_prompts)
        # Collect golden cases once; they are typically not in eval_cases so must be
        # evaluated separately before the golden-failure quality gate can fire.
        _gate_golden_cases = [c for c in cases if c.split == "golden"]
        _gate_golden_eval_map: dict[str, EvalResult] = {}

        for strategy in DISTILLED_STRATEGIES:
            strategy_eval = eval_map.get(strategy)
            if strategy_eval is None:
                continue
            gate_stage_complete = await is_stage_complete(
                session, job_id, PipelineStage.CHECKING_QUALITY_GATES, strategy=strategy
            )
            if not gate_stage_complete:
                await update_job_status(
                    session, job_id, status="running", stage=PipelineStage.CHECKING_QUALITY_GATES
                )
            db_synth = await get_selected_synthesized_rules_for_job(session, job_id, strategy)
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            prompt_text, _ = compile_prompt(task, schemas, strategy)
            compiled_prompts[strategy] = prompt_text
            token_counts[strategy] = count_tokens_approx(prompt_text)
            if _gate_golden_cases:
                set_tracking_context(
                    ModelCallContext(
                        job_id=job_id,
                        stage=PipelineStage.CHECKING_QUALITY_GATES,
                        role="student",
                        provider_profile=student_profile,
                        provider=student_config.provider,
                        model=student_config.model,
                        strategy=strategy,
                    ),
                    collector,
                )
                try:
                    _gate_golden_eval_map[strategy] = await evaluate_prompt(
                        system_prompt=prompt_text,
                        cases=_gate_golden_cases,
                        task=task,
                        chat_client=student_chat,
                        config=student_config,
                        strategy=f"{strategy}_golden",
                        split="golden",
                    )
                except Exception as _golden_exc:
                    logger.warning(
                        "quality_gate_golden_eval_failed",
                        job_id=job_id,
                        strategy=strategy,
                        error=str(_golden_exc),
                    )
            gate = check_quality_gates(
                strategy=strategy,
                distilled_eval=strategy_eval,
                baseline_eval=baseline_scaffold_eval or baseline_eval,
                cases=cases,
                task_mode=task.task_mode,
                task_gates=task.quality_gates,
                settings_defaults=settings.default_quality_gate,
                prompt_token_count=token_counts[strategy],
                golden_eval=_gate_golden_eval_map.get(strategy),
            )
            gate_map[strategy] = gate
            logger.info(
                "quality_gate_result",
                job_id=job_id,
                strategy=strategy,
                passed=gate.passed,
                violations=gate.violations,
                malformed_output_rate=gate.malformed_output_rate,
                metric_delta=gate.metric_delta,
                golden_failures=gate.golden_failures,
            )
            if not gate_stage_complete:
                await mark_stage_complete(
                    session, job_id, PipelineStage.CHECKING_QUALITY_GATES, strategy=strategy
                )

        for strategy in (
            *BASELINE_FEW_SHOT_STRATEGIES,
            *EMBEDDING_BASELINE_STRATEGIES,
            RETRIEVAL_FEW_SHOT_STRATEGY,
        ):
            strategy_eval = eval_map.get(strategy)
            if strategy_eval is None:
                continue
            if strategy not in token_counts:
                fallback_tokens = strategy_eval.prompt_token_count or baseline_prompt_tokens
                token_counts[strategy] = fallback_tokens

            gate_stage_complete = await is_stage_complete(
                session,
                job_id,
                PipelineStage.CHECKING_QUALITY_GATES,
                strategy=strategy,
            )
            if not gate_stage_complete:
                await update_job_status(
                    session, job_id, status="running", stage=PipelineStage.CHECKING_QUALITY_GATES
                )

            gate = check_quality_gates(
                strategy=strategy,
                distilled_eval=strategy_eval,
                baseline_eval=baseline_scaffold_eval or baseline_eval,
                cases=cases,
                task_mode=task.task_mode,
                task_gates=task.quality_gates,
                settings_defaults=settings.default_quality_gate,
                prompt_token_count=token_counts[strategy],
            )
            gate_map[strategy] = gate
            logger.info(
                "quality_gate_result",
                job_id=job_id,
                strategy=strategy,
                passed=gate.passed,
                violations=gate.violations,
                malformed_output_rate=gate.malformed_output_rate,
                metric_delta=gate.metric_delta,
                golden_failures=gate.golden_failures,
            )
            if not gate_stage_complete:
                await mark_stage_complete(
                    session,
                    job_id,
                    PipelineStage.CHECKING_QUALITY_GATES,
                    strategy=strategy,
                )

        unavailable_strategies = {
            strategy: str(meta.get("unavailable_reason") or "unavailable")
            for strategy, meta in strategy_metadata.items()
            if meta.get("unavailable")
        }
        comparison = build_strategy_comparison(
            baseline_eval=baseline_scaffold_eval or baseline_eval,
            dbscan_eval=eval_map.get("dbscan"),
            hdbscan_eval=eval_map.get("hdbscan"),
            dbscan_gate=gate_map.get("dbscan"),
            hdbscan_gate=gate_map.get("hdbscan"),
            task_mode=task.task_mode,
            dbscan_token_count=token_counts.get("dbscan", 0),
            hdbscan_token_count=token_counts.get("hdbscan", 0),
            strategy_evals=eval_map,
            strategy_gates=gate_map,
            strategy_prompt_tokens=token_counts,
            baseline_strategy=BASELINE_SCAFFOLD_STRATEGY,
            unavailable_strategies=unavailable_strategies,
        )
        comparison.strategy_metadata.update(strategy_metadata)
        comparison.evaluation_split_warning = ctx.split_policy.fallback_warning
        selected_strategy = comparison.selected_strategy or BASELINE_SCAFFOLD_STRATEGY
        comparison.selected_strategy_id = selected_strategy
        comparison.selected_strategy_family = _strategy_family(selected_strategy)

        best_baseline_strategy_id = _best_strategy_for_family(
            strategy_evals=eval_map,
            strategy_prompt_tokens=token_counts,
            primary_metric=primary_metric,
            family="baseline",
        )
        best_distilled_strategy_id = _best_strategy_for_family(
            strategy_evals=eval_map,
            strategy_prompt_tokens=token_counts,
            primary_metric=primary_metric,
            family="distilled",
        )
        best_by_family: dict[str, str] = {}
        if best_baseline_strategy_id is not None:
            best_by_family["baseline"] = best_baseline_strategy_id
        if best_distilled_strategy_id is not None:
            best_by_family["distilled"] = best_distilled_strategy_id
        for extra_family in ("embedding_baseline", "retrieval_baseline"):
            best_for_family = _best_strategy_for_family(
                strategy_evals=eval_map,
                strategy_prompt_tokens=token_counts,
                primary_metric=primary_metric,
                family=extra_family,
            )
            if best_for_family is not None:
                best_by_family[extra_family] = best_for_family

        comparison.best_baseline_strategy_id = best_baseline_strategy_id
        comparison.best_distilled_strategy_id = best_distilled_strategy_id
        comparison.best_by_family = best_by_family

        baseline_reference_eval = baseline_scaffold_eval or baseline_eval
        selected_eval = eval_map.get(selected_strategy)
        if (
            _is_classification_task_mode(task.task_mode)
            and baseline_reference_eval is not None
            and selected_eval is not None
        ):
            paired_comparison_artifacts, regressed_labels = _build_runtime_paired_comparison(
                eval_cases=eval_cases,
                baseline_eval=baseline_reference_eval,
                candidate_eval=selected_eval,
                baseline_strategy_id=BASELINE_SCAFFOLD_STRATEGY,
                candidate_strategy_id=selected_strategy,
            )
            state.paired_comparison_artifacts = paired_comparison_artifacts
            comparison.paired_comparison = paired_comparison_artifacts.summary

            selected_eval = selected_eval.model_copy(update={"regressed_labels": regressed_labels})
            eval_map[selected_strategy] = selected_eval
            comparison.strategy_evals[selected_strategy] = selected_eval
            if selected_strategy == "dbscan":
                comparison.dbscan_eval = selected_eval
            elif selected_strategy == "hdbscan":
                comparison.hdbscan_eval = selected_eval
            elif selected_strategy == BASELINE_SCAFFOLD_STRATEGY:
                comparison.baseline_eval = selected_eval

        # ── Stage: selecting_strategy ─────────────────────────────────────
        await StrategySelectionService(session).run(ctx, comparison, selected_strategy)

        # ── Stage: analyzing_failures ─────────────────────────────────────
        failure_analysis_result = await FailureAnalysisService(session).run(
            ctx, state, selected_strategy
        )

        # ── Stage: refining_rules ─────────────────────────────────────────
        await RefinementLoopService(session).run(
            ctx, bundle, state, selected_strategy, failure_analysis_result
        )

        # ── Stage: ablating_rules ─────────────────────────────────────────
        ablation_artifact = await RuleAblationService(session).run(
            ctx, bundle, state, selected_strategy
        )

        # ── Stage: optimizing_pruning (two-pass) ─────────────────────────
        pruning_mode_comparison = await PruningOptimizationService(session).run(
            ctx, bundle, state, selected_strategy, ablation_artifact
        )
        if pruning_mode_comparison is not None:
            comparison.pruning_mode_comparison = pruning_mode_comparison

        # ── Stage: evaluating_classroom ───────────────────────────────────
        await ClassroomEvaluationService(session).run(ctx, state, comparison)

        # ── Stage: logging_artifacts ──────────────────────────────────────
        await MlflowLoggingService(session).run(ctx, bundle, state, comparison, selected_strategy)

        # ── Stage: exporting_artifacts ────────────────────────────────────
        await ArtifactExportService(session).run(
            ctx, state, comparison, selected_strategy, ablation_artifact
        )

        await update_job_status(session, job_id, status="completed", stage=PipelineStage.COMPLETED)
        logger.info("pipeline_completed", job_id=job_id)


