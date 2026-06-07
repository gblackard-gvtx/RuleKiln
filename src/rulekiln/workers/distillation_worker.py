"""Full distillation pipeline worker with stage orchestration, resume semantics, and idempotency."""

from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.artifacts.settings_snapshot import write_settings_snapshot
from rulekiln.artifacts.writer import (
    write_baseline_prompt,
    write_baseline_scaffold_prompt,
    write_cases_normalized,
    write_confusion_matrix_csv,
    write_eval_report,
    write_manifest,
    write_mlflow_run_id,
    write_paired_comparison_artifacts,
    write_per_label_metrics_csv,
    write_prompt,
    write_rule_ablation_json,
    write_rule_provenance_json,
    write_rule_provenance_markdown,
    write_selected_prompt,
    write_strategy_comparison,
    write_strategy_eval,
    write_strategy_prompt,
    write_task,
    write_token_cost_summary,
    write_top_confusions_markdown,
)
from rulekiln.benchmarks.reporting import (
    student_eval_summary_from_eval_result,
    write_per_student_artifacts,
)
from rulekiln.config.settings import get_settings
from rulekiln.db.models import (
    MicroRule,
)
from rulekiln.db.repositories.eval_case_results import (
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.db.repositories.jobs import (
    get_rule_clusters_for_job,
    get_selected_synthesized_rules_for_job,
    get_synthesized_rules_for_job,
    is_stage_complete,
    mark_stage_complete,
    set_mlflow_run_id,
    update_job_status,
)
from rulekiln.db.repositories.model_calls import (
    bulk_insert_model_call_events,
    summarize_model_call_events,
    update_job_usage_totals,
)
from rulekiln.db.session import get_session_factory
from rulekiln.integrations.mlflow_tracker import (
    build_demo_eval_metrics,
    build_demo_params,
    build_provider_params,
    build_run_params,
    create_run,
    log_artifacts_dir,
    log_metrics,
    log_params,
    log_token_cost_metrics,
)
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.baseline_strategies import (
    BASELINE_FEW_SHOT_STRATEGY_TO_K,
    BASELINE_SCAFFOLD_STRATEGY,
    EMBEDDING_CENTROID_STRATEGY,
    EMBEDDING_KNN_STRATEGY_TO_K,
    RETRIEVAL_FEW_SHOT_STRATEGY,
    build_few_shot_prompt_with_budget,
    select_deterministic_few_shot_examples,
)
from rulekiln.pipeline.classroom_evaluator import evaluate_classroom as _evaluate_classroom
from rulekiln.pipeline.evaluator import (
    evaluate_prompt,
)
from rulekiln.pipeline.failure_analysis import FailureAnalysisResult, analyze_failures
from rulekiln.pipeline.prompt_compiler import (
    compile_prompt,
    count_tokens_approx,
)
from rulekiln.pipeline.quality_gates import check_quality_gates
from rulekiln.pipeline.rule_provenance import (
    build_cluster_id_by_micro_rule,
    build_provenance_records,
)
from rulekiln.pipeline.rule_pruning import PruningMode, prune_rules
from rulekiln.pipeline.rule_refinement import apply_refinements, refine_rules_with_teacher
from rulekiln.pipeline.statistics import (
    PairedComparisonArtifacts,
)
from rulekiln.pipeline.strategy_selection import build_strategy_comparison
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.providers.tracking import (
    ModelCallContext,
    set_tracking_context,
)
from rulekiln.schemas.classroom import StudentConfig
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    PruningModeComparison,
    PruningModeRow,
    RefinementIterationArtifact,
    RuleAblationArtifact,
    RuleAblationRecord,
    StudentEvalSummary,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask
from rulekiln.workers.error_classification import format_worker_error_message
from rulekiln.workers.pipeline.artifacts_state import PipelineArtifactsState
from rulekiln.workers.pipeline.baseline_eval_service import BaselineEvaluationService
from rulekiln.workers.pipeline.classification_metrics import (
    _build_runtime_paired_comparison,
    _is_classification_task_mode,
    _refresh_eval_classification_statistics,
    _resolve_training_cases,
)
from rulekiln.workers.pipeline.compile_phase_stages import (
    PromptCompilationService,
    PruningStageService,
    StaticConflictReviewService,
)
from rulekiln.workers.pipeline.context_factory import PipelineContextFactory
from rulekiln.workers.pipeline.db_mappers import (
    _build_eval_case_upsert_payload,
    _db_synth_to_schema,
    _eval_case_record_to_schema,
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
    _insert_eval_run_if_missing,
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
from rulekiln.workers.pipeline.paths import (
    _artifact_root,
    _build_manifest_entries,
)
from rulekiln.workers.pipeline.phase_plan import PipelinePhase
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle_factory import ProviderBundleFactory
from rulekiln.workers.pipeline.seeding import _evaluation_bootstrap_seed
from rulekiln.workers.pipeline.strategy_utils import (
    DISTILLED_STRATEGIES,
    _best_strategy_for_family,
    _delta_vs_baseline,
    _primary_score_for_metric,
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
    teacher_config = bundle.teacher_config
    student_config = bundle.student_config
    embedding_config = bundle.embedding_config
    student_chat = bundle.student_chat
    eval_student_id = bundle.eval_student_id
    primary_metric = ctx.primary_metric
    dataset = ctx.dataset_id

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
    baseline_prompt_text = state.baseline_prompt_text
    baseline_prompt_source = state.baseline_prompt_source
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
            loaded = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=strategy,
                split=eval_split,
            )
            if loaded is not None:
                eval_map[strategy] = loaded

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
                await _set_stage(session, job_id, PipelineStage.CHECKING_QUALITY_GATES)
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
                await _set_stage(session, job_id, PipelineStage.CHECKING_QUALITY_GATES)

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
            if not gate_stage_complete:
                await mark_stage_complete(
                    session,
                    job_id,
                    PipelineStage.CHECKING_QUALITY_GATES,
                    strategy=strategy,
                )

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
        # Evaluate every LLM-prompt strategy across all classroom students.
        # Skipped when classroom_config has only one student (anchor-only run).
        _classroom_student_evals: dict[str, dict[str, EvalResult]] = {}
        _all_student_results: dict[str, dict[str, StudentEvalSummary]] = {}
        _classroom_cfg = payload.classroom_config
        if (
            _classroom_cfg is not None
            and len(_classroom_cfg.students) > 1
            and not await is_stage_complete(session, job_id, PipelineStage.EVALUATING_CLASSROOM)
        ):
            await _set_stage(session, job_id, PipelineStage.EVALUATING_CLASSROOM)

            def _classroom_chat_client(student: StudentConfig) -> ChatModelClient:
                cfg = resolve_provider_config(
                    student.provider, student.model, role="student", settings=settings
                )
                return get_chat_client(cfg)

            def _classroom_provider_config(student: StudentConfig) -> ProviderConfig:
                return resolve_provider_config(
                    student.provider, student.model, role="student", settings=settings
                )

            _llm_prompt_strategies = {
                s: p for s, p in compiled_prompts.items() if s not in EMBEDDING_BASELINE_STRATEGIES
            }
            for _cls_strategy, _cls_prompt in _llm_prompt_strategies.items():
                try:
                    _cls_results = await _evaluate_classroom(
                        system_prompt=_cls_prompt,
                        cases=eval_cases,
                        task=task,
                        classroom_config=_classroom_cfg,
                        get_chat_client=_classroom_chat_client,
                        get_provider_config=_classroom_provider_config,
                        strategy=_cls_strategy,
                        split=eval_split,
                        bootstrap_enabled=evaluation_bootstrap_enabled,
                        bootstrap_iterations=evaluation_bootstrap_iterations,
                        bootstrap_seed=_evaluation_bootstrap_seed(
                            job_id=job_id,
                            strategy=_cls_strategy,
                            split=eval_split,
                            seed_offset=evaluation_bootstrap_seed_offset,
                        ),
                    )
                    _classroom_student_evals[_cls_strategy] = _cls_results
                    _all_student_results[_cls_strategy] = {
                        sid: student_eval_summary_from_eval_result(r, sid)
                        for sid, r in _cls_results.items()
                    }
                except Exception as _cls_exc:
                    logger.warning(
                        "classroom_eval_strategy_failed",
                        job_id=job_id,
                        strategy=_cls_strategy,
                        error=str(_cls_exc),
                    )
            if _all_student_results:
                comparison.all_student_results = _all_student_results
            await mark_stage_complete(session, job_id, PipelineStage.EVALUATING_CLASSROOM)

        # ── Late stages ───────────────────────────────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.LOGGING_ARTIFACTS):
            await _set_stage(session, job_id, PipelineStage.LOGGING_ARTIFACTS)
            selected_prompt_text = compiled_prompts.get(selected_strategy)
            if selected_strategy in DISTILLED_STRATEGIES and selected_prompt_text is None:
                selected_db = await get_selected_synthesized_rules_for_job(
                    session, job_id, selected_strategy
                )
                selected_schemas = [_db_synth_to_schema(r) for r in selected_db]
                selected_prompt_text, _ = compile_prompt(task, selected_schemas, selected_strategy)
                compiled_prompts[selected_strategy] = selected_prompt_text

            prompt_hash_source = selected_prompt_text or baseline_variant_prompts.get(
                BASELINE_SCAFFOLD_STRATEGY,
                baseline_prompt_text,
            )
            prompt_hash = sha256(prompt_hash_source.encode("utf-8")).hexdigest()

            mlflow_run_id = create_run(
                settings.mlflow_tracking_uri,
                settings.mlflow_experiment_name,
                job_id=job_id,
                task_id=task.task_id,
                task_name=task.task_name,
            )
            await set_mlflow_run_id(session, job_id, mlflow_run_id)
            artifact_root = _artifact_root(settings.artifact_root, job_id)
            write_mlflow_run_id(artifact_root, mlflow_run_id)

            run_params: dict[str, str] = {
                **build_run_params(
                    job_id=job_id,
                    task_id=task.task_id,
                    strategy=selected_strategy,
                    prompt_hash=prompt_hash,
                ),
                **build_provider_params(payload),
                **build_demo_params(
                    task_id=task.task_id,
                    task_mode=task.task_mode,
                    dataset=dataset,
                    teacher_provider=teacher_config.provider,
                    teacher_model=teacher_config.model,
                    student_provider=student_config.provider,
                    student_model=student_config.model,
                    embedding_model=embedding_config.model,
                    selected_strategy=selected_strategy,
                    primary_metric=primary_metric,
                ),
                "baseline_prompt_source": baseline_prompt_source,
                "baseline_prompt_compiler": task.baseline_prompt_policy.compiler,
            }
            log_params(settings.mlflow_tracking_uri, mlflow_run_id, run_params)

            selected_eval_for_metrics = eval_map.get(selected_strategy)
            if selected_eval_for_metrics is None and selected_strategy == BASELINE_STRATEGY:
                selected_eval_for_metrics = baseline_eval
            selected_gate = gate_map.get(selected_strategy)
            dbscan_eval = eval_map.get("dbscan")
            hdbscan_eval = eval_map.get("hdbscan")
            baseline_reference_eval = baseline_scaffold_eval or baseline_eval
            eval_metrics = build_demo_eval_metrics(
                baseline_macro_f1=(
                    baseline_reference_eval.macro_f1 if baseline_reference_eval else None
                ),
                baseline_accuracy=(
                    baseline_reference_eval.accuracy if baseline_reference_eval else None
                ),
                baseline_malformed_output_rate=(
                    baseline_reference_eval.malformed_output_rate
                    if baseline_reference_eval
                    else None
                ),
                dbscan_macro_f1=dbscan_eval.macro_f1 if dbscan_eval else None,
                dbscan_accuracy=dbscan_eval.accuracy if dbscan_eval else None,
                dbscan_delta_vs_baseline=_delta_vs_baseline(
                    dbscan_eval,
                    baseline_reference_eval,
                    primary_metric,
                ),
                hdbscan_macro_f1=hdbscan_eval.macro_f1 if hdbscan_eval else None,
                hdbscan_accuracy=hdbscan_eval.accuracy if hdbscan_eval else None,
                hdbscan_delta_vs_baseline=_delta_vs_baseline(
                    hdbscan_eval,
                    baseline_reference_eval,
                    primary_metric,
                ),
                selected_primary_score=_primary_score_for_metric(
                    selected_eval_for_metrics,
                    primary_metric,
                ),
                selected_delta_vs_baseline=_delta_vs_baseline(
                    selected_eval_for_metrics,
                    baseline_reference_eval,
                    primary_metric,
                ),
                selected_passed_quality_gates=(
                    selected_gate.passed if selected_gate is not None else False
                ),
            )
            log_metrics(settings.mlflow_tracking_uri, mlflow_run_id, eval_metrics)

            logger.info(
                "mlflow_params_logged",
                job_id=job_id,
                run_id=mlflow_run_id,
                strategy=selected_strategy,
            )
            await mark_stage_complete(session, job_id, PipelineStage.LOGGING_ARTIFACTS)

        if not await is_stage_complete(session, job_id, PipelineStage.EXPORTING_ARTIFACTS):
            await _set_stage(session, job_id, PipelineStage.EXPORTING_ARTIFACTS)
            artifact_root = _artifact_root(settings.artifact_root, job_id)
            written_artifacts = [
                write_task(artifact_root, task),
                write_cases_normalized(artifact_root, cases),
                write_baseline_prompt(artifact_root, baseline_prompt_text),
                write_baseline_scaffold_prompt(artifact_root, baseline_prompt_text),
                write_eval_report(artifact_root, comparison),
                write_strategy_comparison(artifact_root, comparison),
                write_settings_snapshot(artifact_root, settings),
            ]

            for strategy_name, strategy_eval in comparison.strategy_evals.items():
                written_artifacts.append(
                    write_strategy_eval(artifact_root, strategy_name, strategy_eval)
                )

            selected_eval_for_artifacts = comparison.strategy_evals.get(selected_strategy)
            if selected_eval_for_artifacts is not None:
                written_artifacts.append(
                    write_confusion_matrix_csv(
                        artifact_root / "outputs" / "confusion_matrix.csv",
                        selected_eval_for_artifacts.confusion_matrix,
                    )
                )
                written_artifacts.append(
                    write_per_label_metrics_csv(
                        artifact_root / "outputs" / "per_label_metrics.csv",
                        selected_eval_for_artifacts.per_label_metrics,
                    )
                )
                written_artifacts.append(
                    write_top_confusions_markdown(
                        artifact_root / "outputs" / "top_confusions.md",
                        selected_eval_for_artifacts.top_confusions,
                        baseline_strategy_id=BASELINE_SCAFFOLD_STRATEGY,
                        candidate_strategy_id=selected_strategy,
                    )
                )

            if paired_comparison_artifacts is not None:
                written_artifacts.extend(
                    write_paired_comparison_artifacts(
                        artifact_root / "outputs" / "paired_comparison",
                        paired_comparison_artifacts,
                    )
                )

            # ── Rule ablation artifact ───────────────────────────────────
            if ablation_artifact is not None:
                written_artifacts.append(write_rule_ablation_json(artifact_root, ablation_artifact))

            # ── Rule provenance artifact ──────────────────────────────────
            if selected_strategy in DISTILLED_STRATEGIES:
                prov_selected_eval = comparison.strategy_evals.get(selected_strategy)
                prov_db_synth = await get_selected_synthesized_rules_for_job(
                    session, job_id, selected_strategy
                )
                prov_schemas = [_db_synth_to_schema(r) for r in prov_db_synth]
                if prov_schemas:
                    prov_failure_analysis = (
                        analyze_failures(
                            baseline_scaffold_eval or baseline_eval,
                            prov_selected_eval,
                            prov_schemas,
                        )
                        if prov_selected_eval
                        else None
                    )
                    db_clusters = await get_rule_clusters_for_job(
                        session, job_id, selected_strategy
                    )
                    cluster_pairs = [(c.id, list(c.rule_ids)) for c in db_clusters]
                    cluster_id_map = build_cluster_id_by_micro_rule(cluster_pairs)
                    prov_artifact = build_provenance_records(
                        job_id=job_id,
                        strategy_id=selected_strategy,
                        selected_rules=prov_schemas,
                        failure_analysis=prov_failure_analysis,
                        cluster_id_by_micro_rule=cluster_id_map,
                        ablation_artifact=ablation_artifact,
                    )
                    written_artifacts.append(
                        write_rule_provenance_json(artifact_root, prov_artifact)
                    )
                    written_artifacts.append(
                        write_rule_provenance_markdown(artifact_root, prov_artifact)
                    )

            # ── Per-student classroom artifacts ──────────────────────────
            if _classroom_student_evals:
                per_student_root = artifact_root / "classroom"
                for _cls_strategy, _cls_student_evals in _classroom_student_evals.items():
                    written_artifacts.extend(
                        write_per_student_artifacts(
                            per_student_root, _cls_strategy, _cls_student_evals
                        )
                    )

            for strategy in DISTILLED_STRATEGIES:
                prompt_text = compiled_prompts.get(strategy)
                if prompt_text is None:
                    db_synth = await get_selected_synthesized_rules_for_job(
                        session, job_id, strategy
                    )
                    schemas = [_db_synth_to_schema(r) for r in db_synth]
                    prompt_text, _ = compile_prompt(task, schemas, strategy)
                    compiled_prompts[strategy] = prompt_text
                written_artifacts.append(write_prompt(artifact_root, strategy, prompt_text))

            train_cases = _resolve_training_cases(extraction_cases, eval_cases)
            for strategy in BASELINE_FEW_SHOT_STRATEGIES:
                prompt_text = compiled_prompts.get(strategy)
                if prompt_text is None:
                    selected_examples = select_deterministic_few_shot_examples(
                        train_cases,
                        k=BASELINE_FEW_SHOT_STRATEGY_TO_K[strategy],
                    )
                    prompt_text, _, _ = build_few_shot_prompt_with_budget(
                        baseline_prompt=baseline_prompt_text,
                        examples=selected_examples,
                        max_prompt_tokens=task.max_prompt_tokens,
                    )
                    compiled_prompts[strategy] = prompt_text
                written_artifacts.append(
                    write_strategy_prompt(artifact_root, strategy, prompt_text)
                )

            if selected_strategy in compiled_prompts:
                selected_prompt_text = compiled_prompts.get(selected_strategy)
                if selected_prompt_text:
                    written_artifacts.append(
                        write_selected_prompt(artifact_root, selected_prompt_text)
                    )

            records = collector.records
            if records:
                await bulk_insert_model_call_events(session, job_id, records)

            usage_summary = await summarize_model_call_events(session, job_id)

            await update_job_usage_totals(session, job_id, usage_summary)
            written_artifacts.append(write_token_cost_summary(artifact_root, usage_summary))

            mlflow_run_id_path = artifact_root / "exports" / "mlflow_run_id.txt"
            if mlflow_run_id_path.exists():
                written_artifacts.append(mlflow_run_id_path)

            manifest_entries = _build_manifest_entries(artifact_root, written_artifacts)
            manifest_path = write_manifest(artifact_root, manifest_entries)
            manifest_entries_with_manifest = sorted(
                {*manifest_entries, str(manifest_path.relative_to(artifact_root))}
            )
            write_manifest(artifact_root, manifest_entries_with_manifest)

            try:
                if mlflow_run_id_path.exists():
                    mlflow_run_id_for_cost = mlflow_run_id_path.read_text(encoding="utf-8").strip()
                    log_artifacts_dir(
                        settings.mlflow_tracking_uri,
                        mlflow_run_id_for_cost,
                        artifact_root,
                    )
                    log_token_cost_metrics(
                        settings.mlflow_tracking_uri,
                        mlflow_run_id_for_cost,
                        usage_summary,
                    )
            except Exception as exc:
                logger.warning("mlflow_cost_logging_failed", error=str(exc), job_id=job_id)

            await mark_stage_complete(session, job_id, PipelineStage.EXPORTING_ARTIFACTS)

        await update_job_status(session, job_id, status="completed", stage=PipelineStage.COMPLETED)
        logger.info("pipeline_completed", job_id=job_id)


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _set_stage(session: AsyncSession, job_id: str, stage: PipelineStage) -> None:
    await update_job_status(session, job_id, status="running", stage=stage)
    logger.info("stage_started", job_id=job_id, stage=stage)


async def _evaluate_prompt_strategy(
    session: AsyncSession,
    *,
    job_id: str,
    strategy: str,
    split: str,
    student_id: str,
    system_prompt: str,
    cases: list[RuleKilnCase],
    task: RuleKilnTask,
    case_by_id: dict[str, RuleKilnCase],
    chat_client: ChatModelClient,
    config: ProviderConfig,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed_offset: int,
) -> EvalResult:
    existing_rows = await get_eval_case_results(
        session,
        job_id=job_id,
        student_id=student_id,
        strategy=strategy,
        split=split,
    )
    completed_case_results = {
        row.case_id: _eval_case_record_to_schema(row) for row in existing_rows
    }

    async def _persist_strategy_case(case_result: CaseEvalResult) -> None:
        case = case_by_id.get(case_result.case_id)
        if case is None:
            return
        payload_row = _build_eval_case_upsert_payload(
            job_id=job_id,
            student_id=student_id,
            strategy=strategy,
            split=split,
            case=case,
            result=case_result,
        )
        await upsert_eval_case_result(session, payload_row)

    result = await evaluate_prompt(
        system_prompt,
        cases,
        task,
        chat_client,
        config,
        strategy=strategy,
        split=split,
        completed_case_results=completed_case_results,
        on_case_result=_persist_strategy_case,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=_evaluation_bootstrap_seed(
            job_id=job_id,
            strategy=strategy,
            split=split,
            seed_offset=bootstrap_seed_offset,
        ),
    )
    await _insert_eval_run_if_missing(
        session,
        job_id=job_id,
        split=split,
        strategy=strategy,
        result=result,
    )
    return result


async def _run_refinement_loop(
    session: AsyncSession,
    *,
    job_id: str,
    task: RuleKilnTask,
    artifact_root_path: Path,
    failure_analysis_result: FailureAnalysisResult,
    selected_strategy: str,
    current_eval: EvalResult,
    eval_cases: list[RuleKilnCase],
    eval_split: str,
    case_by_id: dict[str, RuleKilnCase],
    golden_cases: list[RuleKilnCase],
    teacher_chat: ChatModelClient,
    teacher_config: ProviderConfig,
    student_chat: ChatModelClient,
    student_config: ProviderConfig,
    primary_metric: str,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    effective_max_iterations: int,
    effective_max_failure_cases: int,
    effective_max_success_cases: int,
    effective_holdout_fraction: float,
    effective_min_metric_delta: float,
    effective_max_golden_failures: int,
) -> tuple[list[SynthesizedRuleSchema] | None, EvalResult | None]:
    """Run the closed-loop conflict resolution loop (paper Phase 3, §3.3).

    Iterates: analyze → refine → re-prune → compile → evaluate until convergence.
    Emits outputs/refinement_iter_{n}.json per iteration.
    Returns (best_rules, best_eval) or (None, None) if no iteration improved the metric.
    The caller is responsible for checking the return and rolling back if needed.

    effective_holdout_fraction reserves that fraction of eval_cases as a held-out
    set that is never used for iterative evaluation, preventing the loop from
    overfitting to the validation split.  0.0 disables holdout (uses all cases).
    """
    outputs_dir = artifact_root_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    db_synth = await get_selected_synthesized_rules_for_job(session, job_id, selected_strategy)
    initial_rules = [_db_synth_to_schema(r) for r in db_synth]
    if not initial_rules:
        return None, None

    # Split eval_cases into refinement set and holdout set (deterministic).
    if effective_holdout_fraction > 0.0 and len(eval_cases) > 1:
        rng_holdout = random.Random(task.refinement_seed)  # noqa: S311
        shuffled = list(eval_cases)
        rng_holdout.shuffle(shuffled)
        holdout_n = max(1, int(len(shuffled) * effective_holdout_fraction))
        refinement_eval_cases = shuffled[holdout_n:]
    else:
        refinement_eval_cases = eval_cases

    baseline_metric = _primary_score_for_metric(current_eval, primary_metric)
    current_metric = baseline_metric
    current_rules = initial_rules
    current_analysis = failure_analysis_result
    best_rules = initial_rules
    best_eval = current_eval
    any_improvement = False

    for iteration in range(effective_max_iterations):
        iter_path = outputs_dir / f"refinement_iter_{iteration}.json"
        if iter_path.exists():
            try:
                prev = RefinementIterationArtifact.model_validate_json(iter_path.read_text())
                current_metric = prev.new_metric
            except Exception as exc:
                logger.warning(
                    "refinement_iter_artifact_load_failed",
                    job_id=job_id,
                    iteration=iteration,
                    error=str(exc),
                )
            logger.info("refinement_iter_resumed", job_id=job_id, iteration=iteration)
            continue

        utility_signals = current_analysis.build_utility_signals()
        if not utility_signals:
            logger.info(
                "refinement_no_utility_signals",
                job_id=job_id,
                iteration=iteration,
            )
            break

        try:
            refinement = await refine_rules_with_teacher(
                current_rules=current_rules,
                failure_analysis_result=current_analysis,
                case_map=case_by_id,
                chat_client=teacher_chat,
                config=teacher_config,
                seed=task.refinement_seed + iteration,
                max_failure_cases=effective_max_failure_cases,
                max_success_cases=effective_max_success_cases,
            )
        except Exception as exc:
            logger.warning(
                "refinement_teacher_failed",
                job_id=job_id,
                iteration=iteration,
                error=str(exc),
            )
            break

        # Stop early if teacher returned no actions (v2) and no revised_rules (v1 compat)
        has_actions = bool(refinement.actions) or bool(refinement.revised_rules)
        if not has_actions:
            logger.info(
                "refinement_no_actions",
                job_id=job_id,
                iteration=iteration,
                stop_reason="no_actions",
            )
            break

        new_rules = apply_refinements(current_rules, refinement)
        if refinement.actions:
            revised_ids = [a.rule_id for a in refinement.actions if a.action != "keep"]
        else:
            revised_ids = [e.rule_id for e in refinement.revised_rules]

        pruning_result = prune_rules(
            new_rules,
            max_rules=task.max_rules,
            max_prompt_tokens=task.max_prompt_tokens,
            min_rule_support_count=task.min_rule_support_count,
            preserve_golden_rules=task.preserve_golden_rules,
            ranking_mode=task.rule_pruning_mode,
            regression_penalty=task.rule_regression_penalty,
            utility_signals=utility_signals,
        )
        if not pruning_result.selected:
            logger.warning("refinement_pruning_empty", job_id=job_id, iteration=iteration)
            break

        iter_strategy = f"{selected_strategy}_refine_{iteration}"
        try:
            new_prompt, _ = compile_prompt(task, pruning_result.selected, iter_strategy)
        except Exception as exc:
            logger.warning(
                "refinement_compile_failed",
                job_id=job_id,
                iteration=iteration,
                error=str(exc),
            )
            break

        try:
            new_eval = await evaluate_prompt(
                system_prompt=new_prompt,
                cases=refinement_eval_cases,
                task=task,
                chat_client=student_chat,
                config=student_config,
                strategy=iter_strategy,
                split=eval_split,
                bootstrap_enabled=bootstrap_enabled,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=_evaluation_bootstrap_seed(
                    job_id=job_id,
                    strategy=iter_strategy,
                    split=eval_split,
                    seed_offset=bootstrap_seed,
                ),
            )
        except Exception as exc:
            logger.warning(
                "refinement_eval_failed",
                job_id=job_id,
                iteration=iteration,
                error=str(exc),
            )
            break

        new_metric = _primary_score_for_metric(new_eval, primary_metric)
        improvement = new_metric - current_metric

        # Evaluate golden cases with the candidate prompt and block acceptance on failures.
        golden_failures_count = 0
        if golden_cases:
            try:
                golden_eval = await evaluate_prompt(
                    system_prompt=new_prompt,
                    cases=golden_cases,
                    task=task,
                    chat_client=student_chat,
                    config=student_config,
                    strategy=f"{selected_strategy}_golden_{iteration}",
                    split="golden",
                )
                golden_failures_count = sum(1 for r in golden_eval.case_results if not r.passed)
            except Exception as exc:
                logger.warning(
                    "refinement_golden_eval_failed",
                    job_id=job_id,
                    iteration=iteration,
                    error=str(exc),
                )

        if golden_failures_count > effective_max_golden_failures:
            stop_reason = "golden_failure"
        elif new_metric < current_metric:
            stop_reason = "regression"
        elif improvement < effective_min_metric_delta:
            stop_reason = "converged"
        else:
            stop_reason = "continue"

        iter_artifact = RefinementIterationArtifact(
            job_id=job_id,
            iteration=iteration,
            strategy_id=selected_strategy,
            prior_metric=current_metric,
            new_metric=new_metric,
            improvement=improvement,
            revised_rule_ids=revised_ids,
            stop_reason=stop_reason if stop_reason != "continue" else None,
            golden_failures=golden_failures_count,
        )
        iter_path.write_text(iter_artifact.model_dump_json())

        # Provenance artifact: machine-readable record of what changed and why
        from rulekiln.pipeline.failure_analysis import UNATTRIBUTED_RULE_ID

        _failure_case_ids = [
            sf.case_id
            for sf in current_analysis.structured_failures
            if sf.failure_class in ("broken", "unchanged_wrong")
            and sf.violated_rule_ids
            and sf.violated_rule_ids != [UNATTRIBUTED_RULE_ID]
        ]
        _success_case_ids = [
            sf.case_id
            for sf in current_analysis.structured_failures
            if sf.failure_class in ("fixed", "unchanged_correct")
        ]
        _provenance_actions = [
            {
                "rule_id": a.rule_id,
                "action": a.action,
                "new_rule_ids": (
                    [a.rule_id]
                    if a.action in ("keep", "modify")
                    else (
                        [f"{a.rule_id}__refined_{i}" for i in range(1, len(a.revised_rules) + 1)]
                        if a.action == "split"
                        else []
                    )
                ),
                "rationale": a.rationale,
            }
            for a in refinement.actions
        ]
        provenance_path = outputs_dir / f"rule_refinement_provenance_{iteration}.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": "rulekiln.rule_refinement_provenance.v1",
                    "job_id": job_id,
                    "iteration": iteration,
                    "anchor_student_id": student_config.model,
                    "implicated_rule_ids": revised_ids,
                    "failure_case_ids": _failure_case_ids,
                    "success_case_ids": _success_case_ids,
                    "actions": _provenance_actions,
                    "prior_metric": current_metric,
                    "new_metric": new_metric,
                    "golden_failures": golden_failures_count,
                    "stop_reason": stop_reason,
                },
                ensure_ascii=False,
            )
        )

        logger.info(
            "refinement_iter_complete",
            job_id=job_id,
            iteration=iteration,
            anchor_student_id=student_config.model,
            failure_count=len(_failure_case_ids),
            implicated_rule_count=len(revised_ids),
            action_count=len(refinement.actions),
            prior_metric=current_metric,
            new_metric=new_metric,
            improvement=improvement,
            golden_failures=golden_failures_count,
            stop_reason=stop_reason,
        )

        if stop_reason in ("regression", "golden_failure"):
            break

        best_rules = pruning_result.selected
        best_eval = new_eval
        current_metric = new_metric
        current_rules = pruning_result.selected
        any_improvement = True

        if stop_reason != "continue":
            break

        current_analysis = analyze_failures(
            None,
            new_eval,
            pruning_result.selected,
            list(case_by_id.values()),
        )

    if not any_improvement:
        return None, None

    final_path = outputs_dir / "refinement_best_rules.json"
    final_path.write_text(
        json.dumps(
            {
                "schema_version": "rulekiln.refinement_best_rules.v1",
                "job_id": job_id,
                "strategy_id": selected_strategy,
                "rule_count": len(best_rules),
                "rules": [r.model_dump(mode="json") for r in best_rules],
            },
            ensure_ascii=False,
        )
    )
    return best_rules, best_eval


async def _run_rule_ablation(
    session: AsyncSession,
    *,
    job_id: str,
    task: RuleKilnTask,
    selected_strategy: str,
    eval_cases: list[RuleKilnCase],
    eval_split: str,
    case_by_id: dict[str, RuleKilnCase],
    student_id: str,
    student_chat: ChatModelClient,
    student_config: ProviderConfig,
    primary_metric: str,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed_offset: int,
    eval_map: dict[str, EvalResult],
) -> RuleAblationArtifact | None:
    """Run leave-one-rule-out ablation for the selected strategy.

    Gated by task.enable_rule_ablation and case count threshold.
    Returns None when ablation is skipped.
    """
    if not task.enable_rule_ablation:
        return None
    if len(eval_cases) > task.small_run_case_threshold:
        logger.info(
            "ablation_skipped_case_threshold",
            job_id=job_id,
            case_count=len(eval_cases),
            threshold=task.small_run_case_threshold,
        )
        return None
    if selected_strategy not in DISTILLED_STRATEGIES:
        return None

    db_synth = await get_selected_synthesized_rules_for_job(session, job_id, selected_strategy)
    selected_schemas = [_db_synth_to_schema(r) for r in db_synth]
    if not selected_schemas:
        return None

    full_eval = eval_map.get(selected_strategy)
    full_score = _primary_score_for_metric(full_eval, primary_metric)
    candidates = selected_schemas[: task.max_ablation_rules]

    records: list[RuleAblationRecord] = []
    for rule in candidates:
        ablation_strategy_id = f"ablation_without_{rule.id}"
        if await is_stage_complete(session, job_id, ablation_strategy_id):
            # Reload previously persisted result from eval_map if present
            prior = eval_map.get(ablation_strategy_id)
            if prior is not None:
                ablation_score = _primary_score_for_metric(prior, primary_metric)
                delta = ablation_score - full_score
                changed = sum(
                    1
                    for br in (full_eval.case_results if full_eval else [])
                    for ar in (prior.case_results if prior else [])
                    if br.case_id == ar.case_id and br.passed != ar.passed
                )
                classification = _ablation_classify(delta, changed, task.ablation_min_changed_cases)
                records.append(
                    RuleAblationRecord(
                        rule_id=rule.id,
                        topic=rule.topic,
                        classification=classification,
                        metric_delta_without_rule=delta,
                        changed_cases=changed,
                        primary_metric=primary_metric,
                    )
                )
            continue

        reduced_rules = [r for r in selected_schemas if r.id != rule.id]
        try:
            reduced_prompt, _ = compile_prompt(task, reduced_rules, ablation_strategy_id)
        except Exception as exc:
            logger.warning(
                "ablation_compile_failed", job_id=job_id, rule_id=rule.id, error=str(exc)
            )
            records.append(
                RuleAblationRecord(
                    rule_id=rule.id,
                    topic=rule.topic,
                    classification="inconclusive",
                    primary_metric=primary_metric,
                    error=f"compile_failed: {exc}",
                )
            )
            await mark_stage_complete(session, job_id, ablation_strategy_id)
            continue

        try:
            ablation_eval = await _evaluate_prompt_strategy(
                session,
                job_id=job_id,
                strategy=ablation_strategy_id,
                split=eval_split,
                student_id=student_id,
                system_prompt=reduced_prompt,
                cases=eval_cases,
                task=task,
                case_by_id=case_by_id,
                chat_client=student_chat,
                config=student_config,
                bootstrap_enabled=bootstrap_enabled,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed_offset=bootstrap_seed_offset,
            )
        except Exception as exc:
            logger.warning("ablation_eval_failed", job_id=job_id, rule_id=rule.id, error=str(exc))
            records.append(
                RuleAblationRecord(
                    rule_id=rule.id,
                    topic=rule.topic,
                    classification="inconclusive",
                    primary_metric=primary_metric,
                    error=f"eval_failed: {exc}",
                )
            )
            await mark_stage_complete(session, job_id, ablation_strategy_id)
            continue

        eval_map[ablation_strategy_id] = ablation_eval
        ablation_score = _primary_score_for_metric(ablation_eval, primary_metric)
        delta = ablation_score - full_score

        full_case_map = {r.case_id: r.passed for r in (full_eval.case_results if full_eval else [])}
        changed = sum(
            1
            for ar in ablation_eval.case_results
            if full_case_map.get(ar.case_id, ar.passed) != ar.passed
        )
        classification = _ablation_classify(delta, changed, task.ablation_min_changed_cases)
        records.append(
            RuleAblationRecord(
                rule_id=rule.id,
                topic=rule.topic,
                classification=classification,
                metric_delta_without_rule=delta,
                changed_cases=changed,
                primary_metric=primary_metric,
            )
        )
        await mark_stage_complete(session, job_id, ablation_strategy_id)

    return RuleAblationArtifact(
        job_id=job_id,
        strategy_id=selected_strategy,
        primary_metric=primary_metric,
        records=records,
    )


async def _run_two_pass_optimizer(
    session: AsyncSession,
    *,
    job_id: str,
    task: RuleKilnTask,
    selected_strategy: str,
    eval_cases: list[RuleKilnCase],
    eval_split: str,
    case_by_id: dict[str, RuleKilnCase],
    student_id: str,
    student_chat: ChatModelClient,
    student_config: ProviderConfig,
    primary_metric: str,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed_offset: int,
    eval_map: dict[str, EvalResult],
    ablation_artifact: RuleAblationArtifact | None,
) -> PruningModeComparison | None:
    """Run pass-2 optimizer (utility / utility_per_token) and produce comparison rows.

    Returns None when the configured mode is support_count (no pass-2 needed).
    """
    configured_mode: PruningMode = getattr(task, "rule_pruning_mode", "support_count")
    if configured_mode == "support_count":
        # Still build a single-row comparison for the report
        pass1_eval = eval_map.get(selected_strategy)
        pass1_score = _primary_score_for_metric(pass1_eval, primary_metric)
        pass1_tokens = (
            pass1_eval.prompt_token_count if pass1_eval and pass1_eval.prompt_token_count else 0
        )
        db_synth = await get_selected_synthesized_rules_for_job(session, job_id, selected_strategy)
        return PruningModeComparison(
            selected_mode="support_count",
            rows=[
                PruningModeRow(
                    mode="support_count",
                    strategy_id=selected_strategy,
                    rule_count=len(db_synth),
                    prompt_tokens=pass1_tokens,
                    primary_metric=primary_metric,
                    score=pass1_score if pass1_score else None,
                    delta_vs_support_count=0.0,
                    evaluated=True,
                )
            ],
        )

    if selected_strategy not in DISTILLED_STRATEGIES:
        return None

    # Build utility signals from ablation (preferred) or provenance fallback
    utility_signals: dict[str, tuple[int, int]] = {}
    if ablation_artifact is not None:
        pass1_eval = eval_map.get(selected_strategy)
        for rec in ablation_artifact.records:
            if rec.classification == "helpful":
                # Ablation said removing it made things worse → it has utility
                # Use changed_cases as a proxy for fixed_count
                fixed = abs(int(rec.changed_cases or 0))
                utility_signals[rec.rule_id] = (fixed, 0)
            elif rec.classification == "harmful":
                broken = abs(int(rec.changed_cases or 0))
                utility_signals[rec.rule_id] = (0, broken)
    # Fallback: use support_count if no ablation
    if not utility_signals:
        db_synth_all = await get_synthesized_rules_for_job(session, job_id, selected_strategy)
        for sr in db_synth_all:
            if not sr.is_pruned:
                utility_signals.setdefault(sr.id, (sr.support_count, 0))

    pass1_eval = eval_map.get(selected_strategy)
    pass1_score = _primary_score_for_metric(pass1_eval, primary_metric)
    pass1_tokens = (
        pass1_eval.prompt_token_count if pass1_eval and pass1_eval.prompt_token_count else 0
    )
    db_synth_pass1 = await get_selected_synthesized_rules_for_job(
        session, job_id, selected_strategy
    )
    schemas_all = [_db_synth_to_schema(r) for r in db_synth_pass1]

    rows: list[PruningModeRow] = [
        PruningModeRow(
            mode="support_count",
            strategy_id=selected_strategy,
            rule_count=len(db_synth_pass1),
            prompt_tokens=pass1_tokens,
            primary_metric=primary_metric,
            score=pass1_score if pass1_score else None,
            delta_vs_support_count=0.0,
            evaluated=True,
        )
    ]

    regression_penalty: float = getattr(task, "rule_regression_penalty", 2.0)

    for mode in ("utility", "utility_per_token"):
        mode_strategy_id = f"{selected_strategy}_pruning_{mode}"
        if await is_stage_complete(session, job_id, mode_strategy_id):
            prior = eval_map.get(mode_strategy_id)
            if prior is not None:
                prior_score = _primary_score_for_metric(prior, primary_metric)
                prior_tokens = prior.prompt_token_count or 0
                rows.append(
                    PruningModeRow(
                        mode=mode,  # type: ignore[arg-type]
                        strategy_id=mode_strategy_id,
                        rule_count=len([r for r in (prior.case_results or []) if True]),
                        prompt_tokens=prior_tokens,
                        primary_metric=primary_metric,
                        score=prior_score if prior_score else None,
                        delta_vs_support_count=(
                            (prior_score - pass1_score)
                            if prior_score is not None and pass1_score is not None
                            else None
                        ),
                        evaluated=True,
                    )
                )
            continue

        pruned_result = prune_rules(
            schemas_all,
            max_rules=task.max_rules,
            max_prompt_tokens=task.max_prompt_tokens,
            min_rule_support_count=task.min_rule_support_count,
            preserve_golden_rules=task.preserve_golden_rules,
            ranking_mode=mode,  # type: ignore[arg-type]
            regression_penalty=regression_penalty,
            utility_signals=utility_signals,
        )
        if not pruned_result.selected:
            await mark_stage_complete(session, job_id, mode_strategy_id)
            continue

        try:
            mode_prompt, _ = compile_prompt(task, pruned_result.selected, mode_strategy_id)
        except Exception as exc:
            logger.warning("pruning_mode_compile_failed", job_id=job_id, mode=mode, error=str(exc))
            await mark_stage_complete(session, job_id, mode_strategy_id)
            continue

        try:
            mode_eval = await _evaluate_prompt_strategy(
                session,
                job_id=job_id,
                strategy=mode_strategy_id,
                split=eval_split,
                student_id=student_id,
                system_prompt=mode_prompt,
                cases=eval_cases,
                task=task,
                case_by_id=case_by_id,
                chat_client=student_chat,
                config=student_config,
                bootstrap_enabled=bootstrap_enabled,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed_offset=bootstrap_seed_offset,
            )
        except Exception as exc:
            logger.warning("pruning_mode_eval_failed", job_id=job_id, mode=mode, error=str(exc))
            await mark_stage_complete(session, job_id, mode_strategy_id)
            continue

        eval_map[mode_strategy_id] = mode_eval
        mode_score = _primary_score_for_metric(mode_eval, primary_metric)
        mode_tokens = mode_eval.prompt_token_count or 0
        rows.append(
            PruningModeRow(
                mode=mode,  # type: ignore[arg-type]
                strategy_id=mode_strategy_id,
                rule_count=len(pruned_result.selected),
                prompt_tokens=mode_tokens,
                primary_metric=primary_metric,
                score=mode_score if mode_score else None,
                delta_vs_support_count=(
                    (mode_score - pass1_score)
                    if mode_score is not None and pass1_score is not None
                    else None
                ),
                evaluated=True,
            )
        )
        await mark_stage_complete(session, job_id, mode_strategy_id)

    return PruningModeComparison(
        selected_mode=configured_mode,
        rows=rows,
    )


def _ablation_classify(
    delta: float,
    changed_cases: int,
    min_changed_cases: int,
) -> Literal["helpful", "harmful", "neutral", "inconclusive"]:
    """Classify a leave-one-rule-out ablation result."""
    if changed_cases < min_changed_cases:
        return "inconclusive"
    if delta < -0.005:
        return "helpful"
    if delta > 0.005:
        return "harmful"
    return "neutral"
