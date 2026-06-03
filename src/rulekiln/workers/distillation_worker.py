"""Full distillation pipeline worker with stage orchestration, resume semantics, and idempotency."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.agents.rule_conflict_review import review_rule_for_conflicts
from rulekiln.agents.rule_extraction import extract_rules_for_case
from rulekiln.agents.rule_synthesis import synthesize_cluster
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
from rulekiln.config.settings import get_settings
from rulekiln.db.models import (
    Case,
    EvalCaseResultRecord,
    EvalRun,
    MicroRule,
    PromptVersion,
    RuleCluster,
    SynthesizedRule,
)
from rulekiln.db.repositories.eval_case_results import (
    EvalCaseResultUpsert,
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.db.repositories.jobs import (
    bulk_insert_cases,
    bulk_insert_micro_rules,
    bulk_insert_rule_clusters,
    get_eval_runs_for_job,
    get_micro_rules_for_job,
    get_rule_clusters_for_job,
    get_selected_synthesized_rules_for_job,
    get_synthesized_rules_for_job,
    insert_eval_run,
    insert_prompt_version,
    is_stage_complete,
    mark_prompt_version_selected,
    mark_stage_complete,
    set_mlflow_run_id,
    update_job_status,
    update_synthesized_rule_pruning,
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
    RETRIEVAL_FEW_SHOT_K,
    RETRIEVAL_FEW_SHOT_STRATEGY,
    build_few_shot_prompt_with_budget,
    case_text_for_embedding,
    expected_label,
    predict_with_centroids,
    predict_with_knn,
    resolve_distance_metric,
    select_deterministic_few_shot_examples,
    select_retrieval_examples,
)
from rulekiln.pipeline.clustering import cluster_dbscan, cluster_hdbscan
from rulekiln.pipeline.evaluator import (
    build_case_result_from_label_prediction,
    build_eval_result_from_case_results,
    evaluate_prompt,
    get_primary_metric,
)
from rulekiln.pipeline.failure_analysis import FailureAnalysisResult, analyze_failures
from rulekiln.pipeline.prompt_compiler import (
    compile_baseline_prompt,
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
from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.pipeline.statistics import (
    PairedComparisonArtifacts,
    compute_classification_statistics,
    compute_paired_comparison,
    compute_regressed_labels,
)
from rulekiln.pipeline.strategy_selection import build_strategy_comparison
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.providers.embedding import get_embedding_client
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.providers.tracking import (
    ModelCallCollector,
    ModelCallContext,
    set_tracking_context,
    update_tracking_context,
)
from rulekiln.schemas.classroom import TeacherConfig
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    MicroRuleSchema,
    OutcomeCondition,
    PruningModeComparison,
    PruningModeRow,
    QualityGateResult,
    RefinementIterationArtifact,
    RegressedLabelRow,
    RuleAblationArtifact,
    RuleAblationRecord,
    RuleClusterSchema,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask, TaskMode
from rulekiln.workers.error_classification import format_worker_error_message

logger = get_logger(__name__)

_CASE_ID_DELIMITER = "::"
BASELINE_STRATEGY = "baseline"
DISTILLED_STRATEGIES: tuple[str, str] = ("dbscan", "hdbscan")
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


class PipelineStage(StrEnum):
    CREATED = "created"
    VALIDATING_PROJECT = "validating_project"
    EXTRACTING_RULES = "extracting_rules"
    EMBEDDING_RULES = "embedding_rules"
    CLUSTERING_RULES = "clustering_rules"
    SYNTHESIZING_RULES = "synthesizing_rules"
    REVIEWING_RULE_CONFLICTS = "reviewing_rule_conflicts"
    PRUNING_RULES = "pruning_rules"
    COMPILING_PROMPTS = "compiling_prompts"
    EVALUATING_BASELINE = "evaluating_baseline"
    EVALUATING_DISTILLED = "evaluating_distilled"
    SELECTING_STRATEGY = "selecting_strategy"
    ANALYZING_FAILURES = "analyzing_failures"
    REFINING_RULES = "refining_rules"
    ABLATING_RULES = "ablating_rules"
    OPTIMIZING_PRUNING = "optimizing_pruning"
    CHECKING_QUALITY_GATES = "checking_quality_gates"
    LOGGING_ARTIFACTS = "logging_artifacts"
    EXPORTING_ARTIFACTS = "exporting_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"


PipelinePhase = Literal[
    "full",
    "validate_project",
    "compile_prompts",
    "evaluate_baseline",
    "evaluate_dbscan",
    "evaluate_hdbscan",
    "aggregate_evaluation_report",
]


async def run_distillation_pipeline(
    job_id: str,
    payload: DistillationRequest,
) -> None:
    """Top-level background task driving the full pipeline stage-by-stage."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            await _run(session, job_id, payload)
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
    await _run(session, job_id, payload, phase=phase)


async def _run(
    session: AsyncSession,
    job_id: str,
    payload: DistillationRequest,
    *,
    phase: PipelinePhase = "full",
) -> None:  # noqa: C901
    settings = get_settings()
    task = payload.task
    cases = payload.cases
    case_by_id = {case.id: case for case in cases}
    split_policy = resolve_split_policy(cases)
    extraction_cases = split_policy.extraction_cases
    eval_cases = split_policy.evaluation_cases
    eval_split = split_policy.evaluation_split
    collector = ModelCallCollector()
    evaluation_bootstrap_enabled = settings.evaluation.bootstrap_enabled
    evaluation_bootstrap_iterations = settings.evaluation.bootstrap_iterations
    evaluation_bootstrap_seed_offset = settings.evaluation.bootstrap_seed_offset

    logger.info(
        "split_policy_resolved",
        job_id=job_id,
        split_counts=split_policy.split_counts,
        extraction_split=split_policy.extraction_split,
        extraction_case_count=len(extraction_cases),
        evaluation_split=eval_split,
        evaluation_case_count=len(eval_cases),
    )
    if split_policy.fallback_warning is not None:
        logger.warning(
            "evaluation_split_fallback",
            job_id=job_id,
            warning=split_policy.fallback_warning,
            split_counts=split_policy.split_counts,
        )

    run_validate = phase in {"full", "validate_project", "compile_prompts"}
    run_compile_phase = phase in {"full", "compile_prompts"}
    run_baseline_eval = phase in {"full", "evaluate_baseline"}
    run_aggregate = phase in {"full", "aggregate_evaluation_report"}

    student_profile = payload.student.provider_profile
    embedding_profile = payload.embedding.provider_profile
    judge_route = payload.judge or payload.teacher
    judge_profile = judge_route.provider_profile

    # ── Stage: validating_project ──────────────────────────────────────────
    if run_validate and not await is_stage_complete(
        session, job_id, PipelineStage.VALIDATING_PROJECT
    ):
        await _set_stage(session, job_id, PipelineStage.VALIDATING_PROJECT)
        await bulk_insert_cases(session, [_to_db_case(job_id, c) for c in cases])
        await mark_stage_complete(session, job_id, PipelineStage.VALIDATING_PROJECT)

    if phase == "validate_project":
        return

    teacher_config = resolve_provider_config(
        payload.teacher.provider_profile,
        payload.teacher.model,
        role="teacher",
        settings=settings,
    )
    student_config = resolve_provider_config(
        payload.student.provider_profile,
        payload.student.model,
        role="student",
        settings=settings,
    )
    embedding_config = resolve_provider_config(
        payload.embedding.provider_profile,
        payload.embedding.model,
        role="embedding",
        settings=settings,
    )
    # judge falls back to teacher when not explicitly specified
    judge_config = resolve_provider_config(
        judge_profile,
        judge_route.model,
        role="judge",
        settings=settings,
    )
    teacher_chat = get_chat_client(teacher_config)
    student_chat = get_chat_client(student_config)
    embedding_client = get_embedding_client(embedding_config)
    judge_chat = get_chat_client(judge_config)
    eval_student_id = student_config.model
    # Per-phase teacher configs — resolved after teacher_chat is available.
    tc: TeacherConfig | None = payload.teacher_config
    extraction_teacher_config = (
        resolve_provider_config(
            tc.for_phase("instruction_extraction").provider,
            tc.for_phase("instruction_extraction").model,
            role="teacher",
            settings=settings,
        )
        if tc is not None
        else teacher_config
    )
    synthesis_teacher_config = (
        resolve_provider_config(
            tc.for_phase("cluster_consolidation").provider,
            tc.for_phase("cluster_consolidation").model,
            role="teacher",
            settings=settings,
        )
        if tc is not None
        else teacher_config
    )
    refinement_teacher_config = (
        resolve_provider_config(
            tc.for_phase("conflict_resolution").provider,
            tc.for_phase("conflict_resolution").model,
            role="teacher",
            settings=settings,
        )
        if tc is not None
        else teacher_config
    )
    extraction_teacher_chat = (
        get_chat_client(extraction_teacher_config) if tc is not None else teacher_chat
    )
    synthesis_teacher_chat = (
        get_chat_client(synthesis_teacher_config) if tc is not None else teacher_chat
    )
    refinement_teacher_chat = (
        get_chat_client(refinement_teacher_config) if tc is not None else teacher_chat
    )
    primary_metric = _resolve_primary_metric(payload, task.task_mode)
    dataset = _build_dataset_identifier(cases)

    if run_compile_phase:
        # ── Stage: extracting_rules ───────────────────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.EXTRACTING_RULES):
            await _set_stage(session, job_id, PipelineStage.EXTRACTING_RULES)
            existing_micro_rules = await get_micro_rules_for_job(session, job_id)
            extracted_case_ids: set[str] = {
                _payload_case_id_from_db_case_id(job_id, db_rule.case_id)
                for db_rule in existing_micro_rules
            }
            inserted_rule_count = 0
            skipped_case_count = 0
            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.EXTRACTING_RULES,
                    role="teacher",
                    provider_profile=extraction_teacher_config.profile_name,
                    provider=extraction_teacher_config.provider,
                    model=extraction_teacher_config.model,
                ),
                collector,
            )
            for case in extraction_cases:
                case_marker = _extracting_case_marker(case.id)
                if case.id in extracted_case_ids or await is_stage_complete(
                    session,
                    job_id,
                    PipelineStage.EXTRACTING_RULES,
                    artifact_type=case_marker,
                ):
                    skipped_case_count += 1
                    continue

                update_tracking_context(case_id=case.id)
                extraction = await extract_rules_for_case(
                    task, case, extraction_teacher_chat, extraction_teacher_config
                )
                case_micro_rules: list[MicroRule] = []
                for rule in extraction.rules:
                    case_micro_rules.append(
                        MicroRule(
                            id=str(uuid.uuid4()),
                            job_id=job_id,
                            case_id=_db_case_id(job_id, case.id),
                            topic=rule.topic,
                            condition=rule.condition,
                            expected_outcome=rule.expected_outcome,
                            output_path=rule.output_path,
                            rationale_summary=rule.rationale_summary,
                            rule_type=rule.rule_type,
                            positive_cues=rule.positive_cues,
                            negative_cues=rule.negative_cues,
                        )
                    )
                if case_micro_rules:
                    await bulk_insert_micro_rules(session, case_micro_rules)
                    inserted_rule_count += len(case_micro_rules)

                await mark_stage_complete(
                    session,
                    job_id,
                    PipelineStage.EXTRACTING_RULES,
                    artifact_type=case_marker,
                )
                extracted_case_ids.add(case.id)

            await mark_stage_complete(session, job_id, PipelineStage.EXTRACTING_RULES)
            logger.info(
                "rules_extracted",
                job_id=job_id,
                count=inserted_rule_count,
                skipped_cases=skipped_case_count,
            )

        db_micro_rules = await get_micro_rules_for_job(session, job_id)
        rule_ids = [r.id for r in db_micro_rules]
        rule_texts = [f"{r.topic}: {r.condition} → {r.expected_outcome}" for r in db_micro_rules]

        # ── Stage: embedding_rules ────────────────────────────────────────
        embeddings: list[list[float]] = []
        if rule_texts:
            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.EMBEDDING_RULES,
                    role="embedding",
                    provider_profile=embedding_profile,
                    provider=embedding_config.provider,
                    model=embedding_config.model,
                ),
                collector,
            )
            embedding_result = await embedding_client.embed_texts(
                texts=rule_texts, config=embedding_config
            )
            embeddings = embedding_result.embeddings
        if not await is_stage_complete(session, job_id, PipelineStage.EMBEDDING_RULES):
            await _set_stage(session, job_id, PipelineStage.EMBEDDING_RULES)
            await mark_stage_complete(session, job_id, PipelineStage.EMBEDDING_RULES)

        # ── Stage: clustering_rules ───────────────────────────────────────
        dbscan_clusters = cluster_dbscan(rule_ids, embeddings) if rule_ids else []
        hdbscan_clusters = cluster_hdbscan(rule_ids, embeddings) if rule_ids else []
        if not await is_stage_complete(session, job_id, PipelineStage.CLUSTERING_RULES):
            await _set_stage(session, job_id, PipelineStage.CLUSTERING_RULES)
            db_clusters = [_to_db_cluster(job_id, c) for c in dbscan_clusters + hdbscan_clusters]
            await bulk_insert_rule_clusters(session, db_clusters)
            await mark_stage_complete(session, job_id, PipelineStage.CLUSTERING_RULES)

        # ── Stage: synthesizing_rules ─────────────────────────────────────
        rule_map: dict[str, MicroRule] = {r.id: r for r in db_micro_rules}
        for strategy, clusters in [
            (DISTILLED_STRATEGIES[0], dbscan_clusters),
            (DISTILLED_STRATEGIES[1], hdbscan_clusters),
        ]:
            if await is_stage_complete(
                session, job_id, PipelineStage.SYNTHESIZING_RULES, strategy=strategy
            ):
                continue
            await _set_stage(session, job_id, PipelineStage.SYNTHESIZING_RULES)
            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.SYNTHESIZING_RULES,
                    role="teacher",
                    provider_profile=synthesis_teacher_config.profile_name,
                    provider=synthesis_teacher_config.provider,
                    model=synthesis_teacher_config.model,
                    strategy=strategy,
                ),
                collector,
            )
            inserted_rule_count = 0
            processed_cluster_count = 0
            skipped_cluster_count = 0
            for cluster in clusters:
                cluster_marker = _synthesis_cluster_marker(strategy, cluster.rule_ids)
                if await is_stage_complete(
                    session,
                    job_id,
                    PipelineStage.SYNTHESIZING_RULES,
                    strategy=strategy,
                    artifact_type=cluster_marker,
                ):
                    skipped_cluster_count += 1
                    continue

                cluster_micro = [
                    MicroRuleSchema(
                        topic=rule_map[rid].topic,
                        condition=rule_map[rid].condition,
                        expected_outcome=rule_map[rid].expected_outcome,
                        output_path=rule_map[rid].output_path,
                        rationale_summary=rule_map[rid].rationale_summary,
                        rule_type=rule_map[rid].rule_type,
                        positive_cues=list(rule_map[rid].positive_cues or []),
                        negative_cues=list(rule_map[rid].negative_cues or []),
                    )
                    for rid in cluster.rule_ids
                    if rid in rule_map
                ]
                case_ids = list(
                    {rule_map[rid].case_id for rid in cluster.rule_ids if rid in rule_map}
                )
                case_ids = [
                    _payload_case_id_from_db_case_id(job_id, case_id) for case_id in case_ids
                ]
                synthesis = await synthesize_cluster(
                    task,
                    cluster.topic,
                    cluster_micro,
                    case_ids,
                    cluster.rule_ids,
                    synthesis_teacher_chat,
                    synthesis_teacher_config,
                )
                cluster_synth_rules: list[SynthesizedRule] = []
                for rule in synthesis.rules:
                    cluster_synth_rules.append(_synth_to_db(job_id, strategy, rule))

                if cluster_synth_rules:
                    session.add_all(cluster_synth_rules)
                    inserted_rule_count += len(cluster_synth_rules)

                await mark_stage_complete(
                    session,
                    job_id,
                    PipelineStage.SYNTHESIZING_RULES,
                    strategy=strategy,
                    artifact_type=cluster_marker,
                )
                processed_cluster_count += 1

            await mark_stage_complete(
                session, job_id, PipelineStage.SYNTHESIZING_RULES, strategy=strategy
            )
            logger.info(
                "synthesis_done",
                job_id=job_id,
                strategy=strategy,
                processed_clusters=processed_cluster_count,
                skipped_clusters=skipped_cluster_count,
                inserted_rules=inserted_rule_count,
            )

        # ── Stage: reviewing_rule_conflicts (static rule review, pre-eval hygiene) ──
        total_train_cases = len(extraction_cases)
        for strategy in DISTILLED_STRATEGIES:
            if await is_stage_complete(
                session, job_id, PipelineStage.REVIEWING_RULE_CONFLICTS, strategy=strategy
            ):
                continue
            await _set_stage(session, job_id, PipelineStage.REVIEWING_RULE_CONFLICTS)
            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.REVIEWING_RULE_CONFLICTS,
                    role="judge",
                    provider_profile=judge_profile,
                    provider=judge_config.provider,
                    model=judge_config.model,
                    strategy=strategy,
                ),
                collector,
            )
            db_synth = await get_synthesized_rules_for_job(session, job_id, strategy)
            reviewed_rule_count = 0
            skipped_rule_count = 0
            for synth_rule in db_synth:
                review_marker = _conflict_review_rule_marker(strategy, synth_rule.id)
                if await is_stage_complete(
                    session,
                    job_id,
                    PipelineStage.REVIEWING_RULE_CONFLICTS,
                    strategy=strategy,
                    artifact_type=review_marker,
                ):
                    skipped_rule_count += 1
                    continue

                schema = _db_synth_to_schema(synth_rule)
                support = len(schema.source_case_ids)
                ratio = support / total_train_cases if total_train_cases > 0 else 0.0
                golden_backed = bool(
                    task.quality_gates and any(cid in set(schema.source_case_ids) for cid in [])
                )
                token_count = count_tokens_approx(f"{schema.topic} {' '.join(schema.applies_when)}")
                await update_synthesized_rule_pruning(
                    session,
                    synth_rule.id,
                    is_pruned=False,
                    pruning_reason=None,
                    support_count=support,
                    support_ratio=ratio,
                    golden_case_backed=golden_backed,
                    estimated_token_count=token_count,
                )
                cluster_micro = [
                    MicroRuleSchema(
                        topic=rule_map[rid].topic,
                        condition=rule_map[rid].condition,
                        expected_outcome=rule_map[rid].expected_outcome,
                        output_path=rule_map[rid].output_path,
                        rationale_summary=rule_map[rid].rationale_summary,
                        rule_type=rule_map[rid].rule_type,
                        positive_cues=list(rule_map[rid].positive_cues or []),
                        negative_cues=list(rule_map[rid].negative_cues or []),
                    )
                    for rid in (schema.source_micro_rule_ids or [])
                    if rid in rule_map
                ]
                review = await review_rule_for_conflicts(
                    task,
                    schema,
                    cluster_micro,
                    judge_chat,
                    judge_config,
                )
                has_conflicts = review.has_conflicts and review.resolution in ("discard",)
                await session.execute(
                    update(SynthesizedRule)
                    .where(SynthesizedRule.id == synth_rule.id)
                    .values(
                        has_conflicts=has_conflicts,
                        conflict_summary=review.conflict_summary,
                        conflicting_micro_rule_ids=review.conflicting_micro_rule_ids,
                    )
                )
                await mark_stage_complete(
                    session,
                    job_id,
                    PipelineStage.REVIEWING_RULE_CONFLICTS,
                    strategy=strategy,
                    artifact_type=review_marker,
                )
                reviewed_rule_count += 1

            await mark_stage_complete(
                session, job_id, PipelineStage.REVIEWING_RULE_CONFLICTS, strategy=strategy
            )
            logger.info(
                "conflict_review_done",
                job_id=job_id,
                strategy=strategy,
                reviewed=reviewed_rule_count,
                skipped=skipped_rule_count,
            )

        # ── Stage: pruning_rules ──────────────────────────────────────────
        for strategy in DISTILLED_STRATEGIES:
            if await is_stage_complete(
                session, job_id, PipelineStage.PRUNING_RULES, strategy=strategy
            ):
                continue
            await _set_stage(session, job_id, PipelineStage.PRUNING_RULES)
            db_synth = await get_synthesized_rules_for_job(session, job_id, strategy)
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            pruning_result = prune_rules(
                schemas,
                max_rules=task.max_rules,
                max_prompt_tokens=task.max_prompt_tokens,
                min_rule_support_count=task.min_rule_support_count,
                preserve_golden_rules=task.preserve_golden_rules,
            )
            selected_ids = {r.id for r in pruning_result.selected}
            for record in pruning_result.pruned:
                await update_synthesized_rule_pruning(
                    session,
                    record.rule.id,
                    is_pruned=True,
                    pruning_reason=record.reason,
                    support_count=record.rule.support_count,
                    support_ratio=record.rule.support_ratio,
                    golden_case_backed=record.rule.golden_case_backed,
                    estimated_token_count=record.rule.estimated_token_count,
                )
            logger.info(
                "pruning_done",
                job_id=job_id,
                strategy=strategy,
                selected=len(selected_ids),
                pruned=len(pruning_result.pruned),
            )
            await mark_stage_complete(
                session, job_id, PipelineStage.PRUNING_RULES, strategy=strategy
            )

        # ── Stage: compiling_prompts ──────────────────────────────────────
        for strategy in DISTILLED_STRATEGIES:
            if await is_stage_complete(
                session, job_id, PipelineStage.COMPILING_PROMPTS, strategy=strategy
            ):
                continue
            await _set_stage(session, job_id, PipelineStage.COMPILING_PROMPTS)
            db_synth = await get_selected_synthesized_rules_for_job(session, job_id, strategy)
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            prompt_text, prompt_hash = compile_prompt(task, schemas, strategy)
            pv = PromptVersion(
                id=str(uuid.uuid4()),
                job_id=job_id,
                task_id=task.task_id,
                task_name=task.task_name,
                strategy=strategy,
                version="v1",
                system_prompt=prompt_text,
                prompt_hash=prompt_hash,
            )
            await insert_prompt_version(session, pv)
            await mark_stage_complete(
                session, job_id, PipelineStage.COMPILING_PROMPTS, strategy=strategy
            )

    if phase == "compile_prompts":
        await _persist_collector_records(session, job_id, collector)
        return

    # ── Stage: evaluating_baseline ────────────────────────────────────────
    if payload.baseline_prompt:
        baseline_prompt_text = payload.baseline_prompt
        baseline_prompt_source = "provided"
    else:
        baseline_prompt_text = compile_baseline_prompt(task)
        baseline_prompt_source = "compiled"

    train_cases = _resolve_training_cases(extraction_cases, eval_cases)
    distance_metric = _resolve_distance_metric_from_task(task)
    baseline_prompt_tokens = count_tokens_approx(baseline_prompt_text)

    baseline_eval: EvalResult | None = None
    baseline_scaffold_eval: EvalResult | None = None
    baseline_variant_evals: dict[str, EvalResult] = {}
    baseline_variant_prompts: dict[str, str] = {
        BASELINE_SCAFFOLD_STRATEGY: baseline_prompt_text,
    }
    strategy_prompt_tokens: dict[str, int] = {
        BASELINE_SCAFFOLD_STRATEGY: baseline_prompt_tokens,
    }
    strategy_metadata: dict[str, dict[str, str | int | float | bool]] = {
        BASELINE_SCAFFOLD_STRATEGY: {
            "prompt_source": baseline_prompt_source,
        },
        RETRIEVAL_FEW_SHOT_STRATEGY: {
            "configured_k": RETRIEVAL_FEW_SHOT_K,
            "distance_metric": distance_metric,
        },
    }
    for strategy_name, k_value in BASELINE_FEW_SHOT_STRATEGY_TO_K.items():
        selected_examples = select_deterministic_few_shot_examples(train_cases, k=k_value)
        prompt_text, prompt_tokens, used_examples = build_few_shot_prompt_with_budget(
            baseline_prompt=baseline_prompt_text,
            examples=selected_examples,
            max_prompt_tokens=task.max_prompt_tokens,
        )
        baseline_variant_prompts[strategy_name] = prompt_text
        strategy_prompt_tokens[strategy_name] = prompt_tokens
        strategy_metadata[strategy_name] = {
            "configured_k": k_value,
            "used_examples": len(used_examples),
        }

    if run_baseline_eval and not await is_stage_complete(
        session, job_id, PipelineStage.EVALUATING_BASELINE
    ):
        await _set_stage(session, job_id, PipelineStage.EVALUATING_BASELINE)

        # Legacy baseline evaluation is retained for compatibility.
        set_tracking_context(
            ModelCallContext(
                job_id=job_id,
                stage=PipelineStage.EVALUATING_BASELINE,
                role="student",
                provider_profile=student_profile,
                provider=student_config.provider,
                model=student_config.model,
                strategy=BASELINE_STRATEGY,
            ),
            collector,
        )
        baseline_eval = await _evaluate_prompt_strategy(
            session,
            job_id=job_id,
            strategy=BASELINE_STRATEGY,
            split=eval_split,
            student_id=eval_student_id,
            system_prompt=baseline_prompt_text,
            cases=eval_cases,
            task=task,
            case_by_id=case_by_id,
            chat_client=student_chat,
            config=student_config,
            bootstrap_enabled=evaluation_bootstrap_enabled,
            bootstrap_iterations=evaluation_bootstrap_iterations,
            bootstrap_seed_offset=evaluation_bootstrap_seed_offset,
        )
        baseline_eval.prompt_token_count = baseline_prompt_tokens

        baseline_scaffold_eval = baseline_eval.model_copy(
            update={
                "strategy": BASELINE_SCAFFOLD_STRATEGY,
                "prompt_token_count": baseline_prompt_tokens,
            }
        )
        baseline_variant_evals[BASELINE_SCAFFOLD_STRATEGY] = baseline_scaffold_eval
        baseline_variant_prompts[BASELINE_SCAFFOLD_STRATEGY] = baseline_prompt_text
        strategy_prompt_tokens[BASELINE_SCAFFOLD_STRATEGY] = baseline_prompt_tokens
        await _insert_eval_run_if_missing(
            session,
            job_id=job_id,
            split=eval_split,
            strategy=BASELINE_SCAFFOLD_STRATEGY,
            result=baseline_scaffold_eval,
        )

        training_cases = _resolve_training_cases(extraction_cases, eval_cases)
        distance_metric = _resolve_distance_metric_from_task(task)

        for strategy_name, k_value in BASELINE_FEW_SHOT_STRATEGY_TO_K.items():
            selected_examples = select_deterministic_few_shot_examples(training_cases, k=k_value)
            prompt_text, prompt_tokens, used_examples = build_few_shot_prompt_with_budget(
                baseline_prompt=baseline_prompt_text,
                examples=selected_examples,
                max_prompt_tokens=task.max_prompt_tokens,
            )
            baseline_variant_prompts[strategy_name] = prompt_text
            strategy_prompt_tokens[strategy_name] = prompt_tokens
            strategy_metadata[strategy_name] = {
                "configured_k": k_value,
                "used_examples": len(used_examples),
            }
            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.EVALUATING_BASELINE,
                    role="student",
                    provider_profile=student_profile,
                    provider=student_config.provider,
                    model=student_config.model,
                    strategy=strategy_name,
                ),
                collector,
            )
            few_shot_eval = await _evaluate_prompt_strategy(
                session,
                job_id=job_id,
                strategy=strategy_name,
                split=eval_split,
                student_id=eval_student_id,
                system_prompt=prompt_text,
                cases=eval_cases,
                task=task,
                case_by_id=case_by_id,
                chat_client=student_chat,
                config=student_config,
                bootstrap_enabled=evaluation_bootstrap_enabled,
                bootstrap_iterations=evaluation_bootstrap_iterations,
                bootstrap_seed_offset=evaluation_bootstrap_seed_offset,
            )
            few_shot_eval.prompt_token_count = prompt_tokens
            baseline_variant_evals[strategy_name] = few_shot_eval

        labeled_train_cases = [case for case in training_cases if expected_label(case) is not None]
        if labeled_train_cases:
            train_texts = [case_text_for_embedding(case) for case in labeled_train_cases]
            train_labels = [
                expected_label(case) or ""  # expected_label is checked above
                for case in labeled_train_cases
            ]

            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.EVALUATING_BASELINE,
                    role="embedding",
                    provider_profile=embedding_profile,
                    provider=embedding_config.provider,
                    model=embedding_config.model,
                    strategy=EMBEDDING_CENTROID_STRATEGY,
                ),
                collector,
            )
            train_embedding_result = await embedding_client.embed_texts(
                texts=train_texts,
                config=embedding_config,
            )
            train_embeddings = train_embedding_result.embeddings

            set_tracking_context(
                ModelCallContext(
                    job_id=job_id,
                    stage=PipelineStage.EVALUATING_BASELINE,
                    role="embedding",
                    provider_profile=embedding_profile,
                    provider=embedding_config.provider,
                    model=embedding_config.model,
                    strategy=EMBEDDING_CENTROID_STRATEGY,
                ),
                collector,
            )
            eval_texts = [case_text_for_embedding(case) for case in eval_cases]
            eval_embedding_result = await embedding_client.embed_texts(
                texts=eval_texts,
                config=embedding_config,
            )
            eval_embeddings = eval_embedding_result.embeddings

            centroid_predictions = predict_with_centroids(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                eval_embeddings=eval_embeddings,
                metric=distance_metric,
            )
            centroid_case_results = [
                build_case_result_from_label_prediction(case, prediction)
                for case, prediction in zip(eval_cases, centroid_predictions, strict=True)
            ]
            await _persist_case_results(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=EMBEDDING_CENTROID_STRATEGY,
                split=eval_split,
                case_by_id=case_by_id,
                case_results=centroid_case_results,
            )
            embedding_centroid_eval = build_eval_result_from_case_results(
                strategy=EMBEDDING_CENTROID_STRATEGY,
                model=embedding_config.model,
                split=eval_split,
                task=task,
                cases=eval_cases,
                case_results=centroid_case_results,
                prompt_token_count=baseline_prompt_tokens,
                bootstrap_enabled=evaluation_bootstrap_enabled,
                bootstrap_iterations=evaluation_bootstrap_iterations,
                bootstrap_seed=_evaluation_bootstrap_seed(
                    job_id=job_id,
                    strategy=EMBEDDING_CENTROID_STRATEGY,
                    split=eval_split,
                    seed_offset=evaluation_bootstrap_seed_offset,
                ),
            )
            await _insert_eval_run_if_missing(
                session,
                job_id=job_id,
                split=eval_split,
                strategy=EMBEDDING_CENTROID_STRATEGY,
                result=embedding_centroid_eval,
            )
            baseline_variant_evals[EMBEDDING_CENTROID_STRATEGY] = embedding_centroid_eval
            strategy_prompt_tokens[EMBEDDING_CENTROID_STRATEGY] = baseline_prompt_tokens
            strategy_metadata[EMBEDDING_CENTROID_STRATEGY] = {
                "distance_metric": distance_metric,
                "train_cases": len(labeled_train_cases),
            }

            for strategy_name, k_value in EMBEDDING_KNN_STRATEGY_TO_K.items():
                knn_predictions = predict_with_knn(
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    eval_embeddings=eval_embeddings,
                    metric=distance_metric,
                    k=k_value,
                    train_ids=[case.id for case in labeled_train_cases],
                )
                knn_case_results = [
                    build_case_result_from_label_prediction(case, prediction)
                    for case, prediction in zip(eval_cases, knn_predictions, strict=True)
                ]
                await _persist_case_results(
                    session,
                    job_id=job_id,
                    student_id=eval_student_id,
                    strategy=strategy_name,
                    split=eval_split,
                    case_by_id=case_by_id,
                    case_results=knn_case_results,
                )
                knn_eval = build_eval_result_from_case_results(
                    strategy=strategy_name,
                    model=embedding_config.model,
                    split=eval_split,
                    task=task,
                    cases=eval_cases,
                    case_results=knn_case_results,
                    prompt_token_count=baseline_prompt_tokens,
                    bootstrap_enabled=evaluation_bootstrap_enabled,
                    bootstrap_iterations=evaluation_bootstrap_iterations,
                    bootstrap_seed=_evaluation_bootstrap_seed(
                        job_id=job_id,
                        strategy=strategy_name,
                        split=eval_split,
                        seed_offset=evaluation_bootstrap_seed_offset,
                    ),
                )
                await _insert_eval_run_if_missing(
                    session,
                    job_id=job_id,
                    split=eval_split,
                    strategy=strategy_name,
                    result=knn_eval,
                )
                baseline_variant_evals[strategy_name] = knn_eval
                strategy_prompt_tokens[strategy_name] = baseline_prompt_tokens
                strategy_metadata[strategy_name] = {
                    "distance_metric": distance_metric,
                    "k": k_value,
                    "train_cases": len(labeled_train_cases),
                }

            retrieval_case_rows = await get_eval_case_results(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                split=eval_split,
            )
            retrieval_case_results: dict[str, CaseEvalResult] = {
                row.case_id: _eval_case_record_to_schema(row) for row in retrieval_case_rows
            }
            retrieval_failure_count = 0
            model_failure_count = 0
            for existing_result in retrieval_case_results.values():
                if existing_result.error and existing_result.error.startswith("retrieval_failure:"):
                    retrieval_failure_count += 1
                elif existing_result.malformed or (
                    existing_result.error and existing_result.error.startswith("model_failure:")
                ):
                    model_failure_count += 1

            retrieval_prompt_token_peak = baseline_prompt_tokens
            for case in eval_cases:
                if case.id in retrieval_case_results:
                    continue

                try:
                    set_tracking_context(
                        ModelCallContext(
                            job_id=job_id,
                            stage=PipelineStage.EVALUATING_BASELINE,
                            role="embedding",
                            provider_profile=embedding_profile,
                            provider=embedding_config.provider,
                            model=embedding_config.model,
                            strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                            case_id=case.id,
                        ),
                        collector,
                    )
                    query_embedding_result = await embedding_client.embed_texts(
                        texts=[case_text_for_embedding(case)],
                        config=embedding_config,
                    )
                    if not query_embedding_result.embeddings:
                        raise RuntimeError("No query embedding returned.")
                    query_embedding = query_embedding_result.embeddings[0]
                except Exception as exc:
                    retrieval_failure = CaseEvalResult(
                        case_id=case.id,
                        score=0.0,
                        passed=False,
                        malformed=False,
                        assertion_scores={},
                        actual_output=None,
                        error=f"retrieval_failure:{type(exc).__name__}",
                    )
                    retrieval_case_results[case.id] = retrieval_failure
                    retrieval_failure_count += 1
                    await _persist_case_results(
                        session,
                        job_id=job_id,
                        student_id=eval_student_id,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        split=eval_split,
                        case_by_id=case_by_id,
                        case_results=[retrieval_failure],
                    )
                    continue

                retrieved_examples = select_retrieval_examples(
                    query_embedding=query_embedding,
                    train_embeddings=train_embeddings,
                    train_cases=labeled_train_cases,
                    metric=distance_metric,
                    k=RETRIEVAL_FEW_SHOT_K,
                    exclude_case_id=case.id,
                )
                if not retrieved_examples:
                    retrieval_failure = CaseEvalResult(
                        case_id=case.id,
                        score=0.0,
                        passed=False,
                        malformed=False,
                        assertion_scores={},
                        actual_output=None,
                        error="retrieval_failure:no_neighbors",
                    )
                    retrieval_case_results[case.id] = retrieval_failure
                    retrieval_failure_count += 1
                    await _persist_case_results(
                        session,
                        job_id=job_id,
                        student_id=eval_student_id,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        split=eval_split,
                        case_by_id=case_by_id,
                        case_results=[retrieval_failure],
                    )
                    continue

                retrieval_prompt, retrieval_prompt_tokens, _ = build_few_shot_prompt_with_budget(
                    baseline_prompt=baseline_prompt_text,
                    examples=retrieved_examples,
                    max_prompt_tokens=task.max_prompt_tokens,
                )
                retrieval_prompt_token_peak = max(
                    retrieval_prompt_token_peak,
                    retrieval_prompt_tokens,
                )

                set_tracking_context(
                    ModelCallContext(
                        job_id=job_id,
                        stage=PipelineStage.EVALUATING_BASELINE,
                        role="student",
                        provider_profile=student_profile,
                        provider=student_config.provider,
                        model=student_config.model,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        case_id=case.id,
                    ),
                    collector,
                )
                single_case_eval = await evaluate_prompt(
                    retrieval_prompt,
                    [case],
                    task,
                    student_chat,
                    student_config,
                    strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                    split=eval_split,
                    bootstrap_enabled=evaluation_bootstrap_enabled,
                    bootstrap_iterations=evaluation_bootstrap_iterations,
                    bootstrap_seed=_evaluation_bootstrap_seed(
                        job_id=job_id,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        split=eval_split,
                        seed_offset=evaluation_bootstrap_seed_offset,
                        case_id=case.id,
                    ),
                )
                if single_case_eval.case_results:
                    case_result = single_case_eval.case_results[0]
                else:
                    case_result = CaseEvalResult(
                        case_id=case.id,
                        score=0.0,
                        passed=False,
                        malformed=True,
                        assertion_scores={},
                        actual_output=None,
                        error="model_failure:no_case_result",
                    )
                if case_result.malformed:
                    model_failure_count += 1
                    if not case_result.error:
                        case_result.error = "model_failure:malformed_output"

                retrieval_case_results[case.id] = case_result
                await _persist_case_results(
                    session,
                    job_id=job_id,
                    student_id=eval_student_id,
                    strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                    split=eval_split,
                    case_by_id=case_by_id,
                    case_results=[case_result],
                )

            ordered_retrieval_results = [
                retrieval_case_results[case.id]
                for case in eval_cases
                if case.id in retrieval_case_results
            ]
            retrieval_eval = build_eval_result_from_case_results(
                strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                model=student_config.model,
                split=eval_split,
                task=task,
                cases=eval_cases,
                case_results=ordered_retrieval_results,
                prompt_token_count=retrieval_prompt_token_peak,
                retrieval_failure_count=retrieval_failure_count,
                model_failure_count=model_failure_count,
                bootstrap_enabled=evaluation_bootstrap_enabled,
                bootstrap_iterations=evaluation_bootstrap_iterations,
                bootstrap_seed=_evaluation_bootstrap_seed(
                    job_id=job_id,
                    strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                    split=eval_split,
                    seed_offset=evaluation_bootstrap_seed_offset,
                ),
            )
            await _insert_eval_run_if_missing(
                session,
                job_id=job_id,
                split=eval_split,
                strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                result=retrieval_eval,
            )
            baseline_variant_evals[RETRIEVAL_FEW_SHOT_STRATEGY] = retrieval_eval
            strategy_prompt_tokens[RETRIEVAL_FEW_SHOT_STRATEGY] = retrieval_prompt_token_peak
            strategy_metadata[RETRIEVAL_FEW_SHOT_STRATEGY] = {
                "k": RETRIEVAL_FEW_SHOT_K,
                "distance_metric": distance_metric,
                "retrieval_failures": retrieval_failure_count,
                "model_failures": model_failure_count,
            }
        else:
            logger.warning(
                "embedding_baselines_skipped",
                job_id=job_id,
                reason="No labeled training cases available for embedding baselines.",
            )

        await mark_stage_complete(session, job_id, PipelineStage.EVALUATING_BASELINE)

    if phase == "evaluate_baseline":
        await _persist_collector_records(session, job_id, collector)
        return

    # ── Stage: evaluating_distilled ───────────────────────────────────────
    eval_map: dict[str, EvalResult] = dict(baseline_variant_evals)
    distilled_strategies: tuple[str, ...]
    if phase == "evaluate_dbscan":
        distilled_strategies = ("dbscan",)
    elif phase == "evaluate_hdbscan":
        distilled_strategies = ("hdbscan",)
    elif phase == "full":
        distilled_strategies = DISTILLED_STRATEGIES
    else:
        distilled_strategies = ()

    for strategy in distilled_strategies:
        if await is_stage_complete(
            session, job_id, PipelineStage.EVALUATING_DISTILLED, strategy=strategy
        ):
            continue
        await _set_stage(session, job_id, PipelineStage.EVALUATING_DISTILLED)
        set_tracking_context(
            ModelCallContext(
                job_id=job_id,
                stage=PipelineStage.EVALUATING_DISTILLED,
                role="student",
                provider_profile=student_profile,
                provider=student_config.provider,
                model=student_config.model,
                strategy=strategy,
            ),
            collector,
        )
        existing_rows = await get_eval_case_results(
            session,
            job_id=job_id,
            student_id=eval_student_id,
            strategy=strategy,
            split=eval_split,
        )
        completed_case_results = {
            row.case_id: _eval_case_record_to_schema(row) for row in existing_rows
        }

        async def _persist_strategy_case(
            case_result: CaseEvalResult,
            *,
            strategy_name: str = strategy,
        ) -> None:
            case = case_by_id.get(case_result.case_id)
            if case is None:
                return
            payload_row = _build_eval_case_upsert_payload(
                job_id=job_id,
                student_id=eval_student_id,
                strategy=strategy_name,
                split=eval_split,
                case=case,
                result=case_result,
            )
            await upsert_eval_case_result(session, payload_row)

        db_synth = await get_selected_synthesized_rules_for_job(session, job_id, strategy)
        schemas = [_db_synth_to_schema(r) for r in db_synth]
        prompt_text, _ = compile_prompt(task, schemas, strategy)
        ev = await evaluate_prompt(
            prompt_text,
            eval_cases,
            task,
            student_chat,
            student_config,
            strategy=strategy,
            split=eval_split,
            completed_case_results=completed_case_results,
            on_case_result=_persist_strategy_case,
            bootstrap_enabled=evaluation_bootstrap_enabled,
            bootstrap_iterations=evaluation_bootstrap_iterations,
            bootstrap_seed=_evaluation_bootstrap_seed(
                job_id=job_id,
                strategy=strategy,
                split=eval_split,
                seed_offset=evaluation_bootstrap_seed_offset,
            ),
        )
        eval_map[strategy] = ev
        existing_eval_runs = await get_eval_runs_for_job(session, job_id)
        has_strategy_eval = any(
            run.strategy == strategy and run.split == eval_split for run in existing_eval_runs
        )
        if not has_strategy_eval:
            await insert_eval_run(session, _eval_to_db(job_id, None, ev))
        await mark_stage_complete(
            session, job_id, PipelineStage.EVALUATING_DISTILLED, strategy=strategy
        )

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
            eval_map = refreshed_eval_map
            if baseline_eval is not None and BASELINE_STRATEGY in eval_map:
                baseline_eval = eval_map[BASELINE_STRATEGY]
            if baseline_scaffold_eval is not None and BASELINE_SCAFFOLD_STRATEGY in eval_map:
                baseline_scaffold_eval = eval_map[BASELINE_SCAFFOLD_STRATEGY]

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
        gate_map: dict[str, QualityGateResult] = {}
        token_counts: dict[str, int] = dict(strategy_prompt_tokens)
        token_counts.setdefault(BASELINE_SCAFFOLD_STRATEGY, baseline_prompt_tokens)
        compiled_prompts: dict[str, str] = dict(baseline_variant_prompts)

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
        comparison.evaluation_split_warning = split_policy.fallback_warning
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
        if not await is_stage_complete(session, job_id, PipelineStage.SELECTING_STRATEGY):
            await _set_stage(session, job_id, PipelineStage.SELECTING_STRATEGY)
            if selected_strategy in DISTILLED_STRATEGIES:
                await mark_prompt_version_selected(session, job_id, selected_strategy)
            logger.info(
                "strategy_selected",
                job_id=job_id,
                strategy=selected_strategy,
                reason=comparison.selection_reason,
            )
            await mark_stage_complete(session, job_id, PipelineStage.SELECTING_STRATEGY)

        # ── Stage: analyzing_failures ─────────────────────────────────────
        failure_analysis_result: FailureAnalysisResult | None = None
        if not await is_stage_complete(session, job_id, PipelineStage.ANALYZING_FAILURES):
            await _set_stage(session, job_id, PipelineStage.ANALYZING_FAILURES)
            selected_eval = eval_map.get(selected_strategy)
            if selected_eval and selected_strategy in DISTILLED_STRATEGIES:
                db_synth = await get_selected_synthesized_rules_for_job(
                    session, job_id, selected_strategy
                )
                selected_schemas = [_db_synth_to_schema(r) for r in db_synth]
                failure_analysis_result = analyze_failures(
                    baseline_scaffold_eval or baseline_eval,
                    selected_eval,
                    selected_schemas,
                    list(case_by_id.values()),
                )
            await mark_stage_complete(session, job_id, PipelineStage.ANALYZING_FAILURES)

        # ── Stage: refining_rules (closed-loop conflict resolution) ──────────
        refined_rules: list[SynthesizedRuleSchema] | None = None
        if not await is_stage_complete(session, job_id, PipelineStage.REFINING_RULES):
            await _set_stage(session, job_id, PipelineStage.REFINING_RULES)
            if (
                task.enable_refinement_loop
                and failure_analysis_result is not None
                and selected_strategy in DISTILLED_STRATEGIES
            ):
                selected_eval_for_loop = eval_map.get(selected_strategy)
                if selected_eval_for_loop is not None:
                    refined_rules, refined_eval = await _run_refinement_loop(
                        session=session,
                        job_id=job_id,
                        task=task,
                        artifact_root_path=_artifact_root(settings.artifact_root, job_id),
                        failure_analysis_result=failure_analysis_result,
                        selected_strategy=selected_strategy,
                        current_eval=selected_eval_for_loop,
                        eval_cases=eval_cases,
                        eval_split=eval_split,
                        case_by_id=case_by_id,
                        teacher_chat=refinement_teacher_chat,
                        teacher_config=refinement_teacher_config,
                        student_chat=student_chat,
                        student_config=student_config,
                        primary_metric=primary_metric,
                        bootstrap_enabled=evaluation_bootstrap_enabled,
                        bootstrap_iterations=evaluation_bootstrap_iterations,
                        bootstrap_seed=evaluation_bootstrap_seed_offset,
                    )
                    if refined_eval is not None:
                        eval_map[selected_strategy] = refined_eval
            await mark_stage_complete(session, job_id, PipelineStage.REFINING_RULES)

        # ── Stage: ablating_rules ─────────────────────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.ABLATING_RULES):
            ablation_artifact = await _run_rule_ablation(
                session,
                job_id=job_id,
                task=task,
                selected_strategy=selected_strategy,
                eval_cases=eval_cases,
                eval_split=eval_split,
                case_by_id=case_by_id,
                student_id=eval_student_id,
                student_chat=student_chat,
                student_config=student_config,
                primary_metric=primary_metric,
                bootstrap_enabled=evaluation_bootstrap_enabled,
                bootstrap_iterations=evaluation_bootstrap_iterations,
                bootstrap_seed_offset=evaluation_bootstrap_seed_offset,
                eval_map=eval_map,
            )
            await mark_stage_complete(session, job_id, PipelineStage.ABLATING_RULES)

        # ── Stage: optimizing_pruning (two-pass) ─────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.OPTIMIZING_PRUNING):
            pruning_mode_comparison = await _run_two_pass_optimizer(
                session,
                job_id=job_id,
                task=task,
                selected_strategy=selected_strategy,
                eval_cases=eval_cases,
                eval_split=eval_split,
                case_by_id=case_by_id,
                student_id=eval_student_id,
                student_chat=student_chat,
                student_config=student_config,
                primary_metric=primary_metric,
                bootstrap_enabled=evaluation_bootstrap_enabled,
                bootstrap_iterations=evaluation_bootstrap_iterations,
                bootstrap_seed_offset=evaluation_bootstrap_seed_offset,
                eval_map=eval_map,
                ablation_artifact=ablation_artifact,
            )
            if pruning_mode_comparison is not None:
                comparison.pruning_mode_comparison = pruning_mode_comparison
            await mark_stage_complete(session, job_id, PipelineStage.OPTIMIZING_PRUNING)

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
                written_artifacts.append(
                    write_rule_ablation_json(artifact_root, ablation_artifact)
                )

            # ── Rule provenance artifact ──────────────────────────────────
            if selected_strategy in DISTILLED_STRATEGIES:
                prov_selected_eval = comparison.strategy_evals.get(selected_strategy)
                prov_db_synth = await get_selected_synthesized_rules_for_job(
                    session, job_id, selected_strategy
                )
                prov_schemas = [_db_synth_to_schema(r) for r in prov_db_synth]
                if prov_schemas:
                    prov_failure_analysis = analyze_failures(
                        baseline_scaffold_eval or baseline_eval,
                        prov_selected_eval,
                        prov_schemas,
                    ) if prov_selected_eval else None
                    db_clusters = await get_rule_clusters_for_job(
                        session, job_id, selected_strategy
                    )
                    cluster_pairs = [
                        (c.id, list(c.rule_ids)) for c in db_clusters
                    ]
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


async def _persist_collector_records(
    session: AsyncSession,
    job_id: str,
    collector: ModelCallCollector,
) -> None:
    if not collector.records:
        return
    await bulk_insert_model_call_events(session, job_id, collector.records)
    await session.commit()


async def _load_eval_result_from_db(
    session: AsyncSession,
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
) -> EvalResult | None:
    eval_runs = await get_eval_runs_for_job(session, job_id)
    run = next((row for row in eval_runs if row.strategy == strategy and row.split == split), None)
    if run is None:
        return None

    case_rows = await get_eval_case_results(
        session,
        job_id=job_id,
        student_id=student_id,
        strategy=strategy,
        split=split,
    )
    case_results = [_eval_case_record_to_schema(row) for row in case_rows]

    confusion_matrix_raw = run.confusion_matrix or {}
    confusion_matrix: dict[str, dict[str, int]] = {
        str(expected): {str(actual): int(count) for actual, count in actuals.items()}
        for expected, actuals in confusion_matrix_raw.items()
    }

    return EvalResult(
        strategy=run.strategy,
        model=run.model,
        split=run.split,
        accuracy=run.accuracy,
        macro_f1=run.macro_f1,
        weighted_case_score=run.weighted_case_score,
        malformed_output_rate=run.malformed_output_rate or 0.0,
        per_outcome_precision=dict(run.per_outcome_precision or {}),
        per_outcome_recall=dict(run.per_outcome_recall or {}),
        confusion_matrix=confusion_matrix,
        case_results=case_results,
    )


async def _set_stage(session: AsyncSession, job_id: str, stage: PipelineStage) -> None:
    await update_job_status(session, job_id, status="running", stage=stage)
    logger.info("stage_started", job_id=job_id, stage=stage)


def _to_db_case(job_id: str, case: RuleKilnCase) -> Case:
    expected_json: dict | None = None  # pyright: ignore[reportMissingTypeArgument]
    expected_text: str | None = None
    if isinstance(case.expected, dict):
        expected_json = case.expected
    elif case.expected is not None:
        expected_text = str(case.expected)
    return Case(
        id=_db_case_id(job_id, case.id),
        job_id=job_id,
        task_mode=case.task_mode,
        split=case.split,
        input_json=case.input,
        expected_json=expected_json,
        expected_text=expected_text,
        evaluation_json=case.evaluation.model_dump(),
        metadata_json=case.metadata,
        weight=case.weight,
    )


def _db_case_id(job_id: str, payload_case_id: str) -> str:
    return f"{job_id}{_CASE_ID_DELIMITER}{payload_case_id}"


def _extracting_case_marker(payload_case_id: str) -> str:
    return f"extracting_case:{payload_case_id}"


def _synthesis_cluster_marker(strategy: str, rule_ids: list[str]) -> str:
    cluster_key = ",".join(sorted(rule_ids))
    digest = sha256(cluster_key.encode("utf-8")).hexdigest()[:16]
    return f"synth_cluster:{strategy}:{digest}"


def _conflict_review_rule_marker(strategy: str, synthesized_rule_id: str) -> str:
    return f"conflict_rule:{strategy}:{synthesized_rule_id}"


def _payload_case_id_from_db_case_id(job_id: str, db_case_id: str) -> str:
    prefix = f"{job_id}{_CASE_ID_DELIMITER}"
    if db_case_id.startswith(prefix):
        return db_case_id[len(prefix) :]
    return db_case_id


def _to_db_cluster(job_id: str, c: RuleClusterSchema) -> RuleCluster:
    return RuleCluster(
        id=str(uuid.uuid4()),
        job_id=job_id,
        strategy=c.strategy,
        topic=c.topic,
        algorithm=c.algorithm,
        rule_ids=c.rule_ids,
        cluster_metadata=c.cluster_metadata,
    )


def _synth_to_db(job_id: str, strategy: str, rule: SynthesizedRuleSchema) -> SynthesizedRule:
    return SynthesizedRule(
        id=str(uuid.uuid4()),
        job_id=job_id,
        strategy=strategy,
        rule_type=rule.rule_type,
        topic=rule.topic,
        applies_when=rule.applies_when,
        outcome_conditions={k: v.model_dump() for k, v in rule.outcome_conditions.items()},
        tie_breakers=rule.tie_breakers,
        priority=rule.priority,
        source_case_ids=rule.source_case_ids,
        source_micro_rule_ids=rule.source_micro_rule_ids,
        has_conflicts=rule.has_conflicts,
        conflict_summary=rule.conflict_summary,
        conflicting_micro_rule_ids=rule.conflicting_micro_rule_ids,
        support_count=rule.support_count,
        support_ratio=rule.support_ratio,
        golden_case_backed=rule.golden_case_backed,
        estimated_token_count=rule.estimated_token_count,
    )


def _db_synth_to_schema(r: SynthesizedRule) -> SynthesizedRuleSchema:
    oc = {k: OutcomeCondition.model_validate(v) for k, v in (r.outcome_conditions or {}).items()}
    return SynthesizedRuleSchema(
        id=r.id,
        rule_type=r.rule_type,
        topic=r.topic,
        applies_when=list(r.applies_when or []),
        outcome_conditions=oc,
        tie_breakers=list(r.tie_breakers or []),
        priority=r.priority,
        source_case_ids=list(r.source_case_ids or []),
        source_micro_rule_ids=list(r.source_micro_rule_ids or []),
        has_conflicts=r.has_conflicts,
        conflict_summary=r.conflict_summary,
        conflicting_micro_rule_ids=list(r.conflicting_micro_rule_ids or []),
        support_count=r.support_count,
        support_ratio=r.support_ratio,
        golden_case_backed=r.golden_case_backed,
        estimated_token_count=r.estimated_token_count,
    )


def _eval_to_db(job_id: str, prompt_version_id: str | None, result: EvalResult) -> EvalRun:
    return EvalRun(
        id=str(uuid.uuid4()),
        job_id=job_id,
        prompt_version_id=prompt_version_id,
        strategy=result.strategy,
        model=result.model,
        split=result.split,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        weighted_case_score=result.weighted_case_score,
        per_outcome_precision=result.per_outcome_precision,
        per_outcome_recall=result.per_outcome_recall,
        malformed_output_rate=result.malformed_output_rate,
        confusion_matrix={k: dict(v) for k, v in result.confusion_matrix.items()},
    )


def _eval_case_record_to_schema(row: EvalCaseResultRecord) -> CaseEvalResult:
    actual_output = row.actual_json
    if actual_output is None and row.raw_output is not None:
        actual_output = row.raw_output

    return CaseEvalResult(
        case_id=row.case_id,
        score=row.case_score,
        passed=row.passed,
        malformed=row.malformed,
        assertion_scores=dict(row.assertion_scores or {}),
        actual_output=actual_output,
        error=row.error_message,
    )


def _build_eval_case_upsert_payload(
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
    case: RuleKilnCase,
    result: CaseEvalResult,
) -> EvalCaseResultUpsert:
    expected_json: dict[str, object] | str | None = (
        case.expected if isinstance(case.expected, (dict, str)) else None
    )

    actual_json = result.actual_output
    raw_output = _actual_output_to_raw_text(result.actual_output)
    invalid_label = _is_invalid_label(case, result.actual_output)

    return EvalCaseResultUpsert(
        job_id=job_id,
        student_id=student_id,
        strategy=strategy,
        split=split,
        case_id=result.case_id,
        expected_json=expected_json,
        actual_json=actual_json,
        raw_output=raw_output,
        assertion_scores=result.assertion_scores,
        passed=result.passed,
        case_score=result.score,
        malformed=result.malformed,
        invalid_label=invalid_label,
        error_type="MalformedOutput" if result.malformed else None,
        error_message=result.error,
    )


def _actual_output_to_raw_text(
    actual_output: dict[str, str | int | float | bool | None] | str | None,
) -> str | None:
    if actual_output is None:
        return None
    if isinstance(actual_output, str):
        return actual_output
    return json.dumps(actual_output, ensure_ascii=False)


def _is_invalid_label(
    case: RuleKilnCase,
    actual_output: dict[str, str | int | float | bool | None] | str | None,
) -> bool:
    if case.task_mode not in {"classification", "routing"}:
        return False

    if actual_output is None:
        return True

    if isinstance(actual_output, dict):
        label = actual_output.get("label")
        return not isinstance(label, str) or label.strip() == ""

    return actual_output.strip() == ""


def _is_classification_task_mode(task_mode: str) -> bool:
    return task_mode in {"classification", "routing"}


def _prediction_label_from_case_result(case_result: CaseEvalResult | None) -> str:
    if case_result is None or case_result.actual_output is None:
        return ""
    if isinstance(case_result.actual_output, dict):
        label_value = case_result.actual_output.get("label")
        if isinstance(label_value, str):
            return label_value.strip()
        return ""
    return case_result.actual_output.strip()


def _case_input_text(case: RuleKilnCase) -> str:
    utterance = case.input.get("utterance")
    if isinstance(utterance, str):
        return utterance

    text_value = case.input.get("text")
    if isinstance(text_value, str):
        return text_value

    return json.dumps(case.input, sort_keys=True, ensure_ascii=False)


def _aligned_classification_columns(
    *,
    eval_cases: list[RuleKilnCase],
    eval_result: EvalResult,
) -> tuple[list[str], list[str], list[str], list[str]]:
    case_results_by_id = {
        case_result.case_id: case_result for case_result in eval_result.case_results
    }

    case_ids: list[str] = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    input_texts: list[str] = []

    for case in eval_cases:
        expected = expected_label(case)
        if expected is None:
            continue
        case_ids.append(case.id)
        expected_labels.append(expected)
        predicted_labels.append(_prediction_label_from_case_result(case_results_by_id.get(case.id)))
        input_texts.append(_case_input_text(case))

    return case_ids, expected_labels, predicted_labels, input_texts


def _refresh_eval_classification_statistics(
    *,
    eval_result: EvalResult,
    eval_cases: list[RuleKilnCase],
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> EvalResult:
    case_ids, expected_labels, predicted_labels, _ = _aligned_classification_columns(
        eval_cases=eval_cases,
        eval_result=eval_result,
    )
    if not case_ids:
        return eval_result

    classification_stats = compute_classification_statistics(
        actual_labels=expected_labels,
        predicted_labels=predicted_labels,
        case_ids=case_ids,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    return eval_result.model_copy(
        update={
            "accuracy": classification_stats.accuracy,
            "accuracy_ci_95": classification_stats.accuracy_ci_95,
            "macro_f1": classification_stats.macro_f1,
            "macro_f1_ci_95": classification_stats.macro_f1_ci_95,
            "per_label_metrics": classification_stats.per_label_metrics,
            "per_outcome_precision": classification_stats.per_outcome_precision,
            "per_outcome_recall": classification_stats.per_outcome_recall,
            "confusion_matrix": classification_stats.confusion_matrix,
            "top_confusions": classification_stats.top_confusions,
        }
    )


def _strategy_family(strategy: str) -> str:
    return "distilled" if strategy in DISTILLED_STRATEGIES else "baseline"


def _best_strategy_for_family(
    *,
    strategy_evals: dict[str, EvalResult],
    strategy_prompt_tokens: dict[str, int],
    primary_metric: str,
    family: Literal["baseline", "distilled"],
) -> str | None:
    candidates = [
        strategy_name
        for strategy_name in strategy_evals
        if _strategy_family(strategy_name) == family
    ]
    if not candidates:
        return None

    def sort_key(strategy_name: str) -> tuple[float, float, int, str]:
        strategy_eval = strategy_evals[strategy_name]
        fallback_token_count = (
            strategy_eval.prompt_token_count if strategy_eval.prompt_token_count is not None else 0
        )
        token_count = strategy_prompt_tokens.get(
            strategy_name,
            fallback_token_count,
        )
        return (
            -_primary_score_for_metric(strategy_eval, primary_metric),
            strategy_eval.malformed_output_rate,
            token_count,
            strategy_name,
        )

    return sorted(candidates, key=sort_key)[0]


def _build_runtime_paired_comparison(
    *,
    eval_cases: list[RuleKilnCase],
    baseline_eval: EvalResult,
    candidate_eval: EvalResult,
    baseline_strategy_id: str,
    candidate_strategy_id: str,
) -> tuple[PairedComparisonArtifacts, list[RegressedLabelRow]]:
    (
        case_ids,
        expected_labels,
        baseline_predictions,
        input_texts,
    ) = _aligned_classification_columns(
        eval_cases=eval_cases,
        eval_result=baseline_eval,
    )
    _, _, candidate_predictions, _ = _aligned_classification_columns(
        eval_cases=eval_cases,
        eval_result=candidate_eval,
    )

    paired_artifacts = compute_paired_comparison(
        case_ids=case_ids,
        actual_labels=expected_labels,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        input_texts=input_texts,
        baseline_strategy_id=baseline_strategy_id,
        candidate_strategy_id=candidate_strategy_id,
    )
    regressed_labels = compute_regressed_labels(
        case_ids=case_ids,
        actual_labels=expected_labels,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
    )
    return paired_artifacts, regressed_labels


def _resolve_primary_metric(payload: DistillationRequest, task_mode: str) -> str:
    if payload.metric and payload.metric.strip():
        return payload.metric.strip()
    valid_task_modes = {
        "classification",
        "summarization",
        "extraction",
        "rubric_review",
        "routing",
        "tool_use",
        "freeform_generation",
        "agent_behavior",
    }
    if task_mode in valid_task_modes:
        return get_primary_metric(cast(TaskMode, task_mode))
    return "weighted_case_score"


def _build_dataset_identifier(cases: list[RuleKilnCase]) -> str:
    sorted_cases = sorted(cases, key=lambda case: case.id)
    case_payloads = [case.model_dump(mode="json") for case in sorted_cases]
    serialized = json.dumps(case_payloads, sort_keys=True, ensure_ascii=False)
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    return f"cases_sha256:{digest}"


def _primary_score_for_metric(result: EvalResult | None, primary_metric: str) -> float:
    if result is None:
        return 0.0
    metric_name = primary_metric.strip().lower()
    if metric_name == "macro_f1":
        return float(result.macro_f1 or 0.0)
    if metric_name == "accuracy":
        return float(result.accuracy or 0.0)
    return float(result.weighted_case_score or 0.0)


def _delta_vs_baseline(
    result: EvalResult | None,
    baseline: EvalResult | None,
    primary_metric: str,
) -> float:
    if result is None or baseline is None:
        return 0.0
    return _primary_score_for_metric(result, primary_metric) - _primary_score_for_metric(
        baseline, primary_metric
    )


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
    teacher_chat: ChatModelClient,
    teacher_config: ProviderConfig,
    student_chat: ChatModelClient,
    student_config: ProviderConfig,
    primary_metric: str,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[list[SynthesizedRuleSchema] | None, EvalResult | None]:
    """Run the closed-loop conflict resolution loop (paper Phase 3, §3.3).

    Iterates: analyze → refine → re-prune → compile → evaluate until convergence.
    Emits outputs/refinement_iter_{n}.json per iteration.
    Returns (best_rules, best_eval) or (None, None) if no iteration improved the metric.
    The caller is responsible for checking the return and rolling back if needed.
    """
    outputs_dir = artifact_root_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    db_synth = await get_selected_synthesized_rules_for_job(session, job_id, selected_strategy)
    initial_rules = [_db_synth_to_schema(r) for r in db_synth]
    if not initial_rules:
        return None, None

    baseline_metric = _primary_score_for_metric(current_eval, primary_metric)
    current_metric = baseline_metric
    current_rules = initial_rules
    current_analysis = failure_analysis_result
    best_rules = initial_rules
    best_eval = current_eval
    any_improvement = False

    for iteration in range(task.refinement_max_iterations):
        iter_path = outputs_dir / f"refinement_iter_{iteration}.json"
        if iter_path.exists():
            try:
                prev = RefinementIterationArtifact.model_validate_json(iter_path.read_text())
                current_metric = prev.new_metric
            except Exception as exc:
                logger.warning(
                    "refinement_iter_artifact_load_failed",
                    job_id=job_id, iteration=iteration, error=str(exc),
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
                max_failure_cases=task.refinement_max_failure_cases,
                max_success_cases=task.refinement_max_success_cases,
            )
        except Exception as exc:
            logger.warning(
                "refinement_teacher_failed",
                job_id=job_id, iteration=iteration, error=str(exc),
            )
            break

        new_rules = apply_refinements(current_rules, refinement)
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
                job_id=job_id, iteration=iteration, error=str(exc),
            )
            break

        try:
            new_eval = await evaluate_prompt(
                system_prompt=new_prompt,
                cases=eval_cases,
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
                job_id=job_id, iteration=iteration, error=str(exc),
            )
            break

        new_metric = _primary_score_for_metric(new_eval, primary_metric)
        improvement = new_metric - current_metric

        if new_metric < current_metric:
            stop_reason = "regression"
        elif improvement < task.refinement_epsilon:
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
        )
        iter_path.write_text(iter_artifact.model_dump_json())
        logger.info(
            "refinement_iter_complete",
            job_id=job_id,
            iteration=iteration,
            prior_metric=current_metric,
            new_metric=new_metric,
            improvement=improvement,
            stop_reason=stop_reason,
        )

        if stop_reason == "regression":
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
            logger.warning(
                "ablation_eval_failed", job_id=job_id, rule_id=rule.id, error=str(exc)
            )
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
                        rule_count=len(
                            [r for r in (prior.case_results or []) if True]
                        ),
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
            logger.warning(
                "pruning_mode_compile_failed", job_id=job_id, mode=mode, error=str(exc)
            )
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
            logger.warning(
                "pruning_mode_eval_failed", job_id=job_id, mode=mode, error=str(exc)
            )
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
) -> Literal['helpful', 'harmful', 'neutral', 'inconclusive']:
    """Classify a leave-one-rule-out ablation result."""
    if changed_cases < min_changed_cases:
        return "inconclusive"
    if delta < -0.005:
        return "helpful"
    if delta > 0.005:
        return "harmful"
    return "neutral"


def _build_manifest_entries(root: Path, paths: Sequence[Path]) -> list[str]:
    entries: set[str] = set()
    for entry in paths:
        if not entry.exists():
            continue
        try:
            entries.add(str(entry.relative_to(root)))
        except ValueError:
            continue
    return sorted(entries)


def _artifact_root(artifact_root_setting: str, job_id: str) -> Path:
    return Path(artifact_root_setting) / job_id


async def _insert_eval_run_if_missing(
    session: AsyncSession,
    *,
    job_id: str,
    split: str,
    strategy: str,
    result: EvalResult,
) -> None:
    existing_eval_runs = await get_eval_runs_for_job(session, job_id)
    has_strategy_eval = any(
        run.strategy == strategy and run.split == split for run in existing_eval_runs
    )
    if not has_strategy_eval:
        await insert_eval_run(session, _eval_to_db(job_id, None, result))


async def _persist_case_results(
    session: AsyncSession,
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
    case_by_id: dict[str, RuleKilnCase],
    case_results: list[CaseEvalResult],
) -> None:
    for case_result in case_results:
        case = case_by_id.get(case_result.case_id)
        if case is None:
            continue
        payload_row = _build_eval_case_upsert_payload(
            job_id=job_id,
            student_id=student_id,
            strategy=strategy,
            split=split,
            case=case,
            result=case_result,
        )
        await upsert_eval_case_result(session, payload_row)


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


def _resolve_distance_metric_from_task(task: RuleKilnTask) -> str:
    raw_metric = task.prompt_scaffold.get("embedding_distance_metric")
    if isinstance(raw_metric, str):
        return resolve_distance_metric(raw_metric)
    return resolve_distance_metric(None)


def _resolve_training_cases(
    extraction_cases: list[RuleKilnCase],
    eval_cases: list[RuleKilnCase],
) -> list[RuleKilnCase]:
    if extraction_cases:
        return extraction_cases
    return eval_cases


def _evaluation_bootstrap_seed(
    *,
    job_id: str,
    strategy: str,
    split: str,
    seed_offset: int,
    case_id: str | None = None,
) -> int:
    raw_seed = f"{job_id}|{strategy}|{split}|{seed_offset}|{case_id or ''}"
    digest = sha256(raw_seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
