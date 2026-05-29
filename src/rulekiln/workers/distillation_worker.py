"""Full distillation pipeline worker with stage orchestration, resume semantics, and idempotency."""

from __future__ import annotations

import json
from hashlib import sha1, sha256
import uuid
from enum import StrEnum
from typing import Literal

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.agents.rule_conflict_review import review_rule_for_conflicts
from rulekiln.agents.rule_extraction import extract_rules_for_case
from rulekiln.agents.rule_synthesis import synthesize_cluster
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
from rulekiln.db.repositories.jobs import (
    bulk_insert_cases,
    bulk_insert_micro_rules,
    bulk_insert_rule_clusters,
    get_eval_runs_for_job,
    get_micro_rules_for_job,
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
from rulekiln.db.repositories.eval_case_results import (
    EvalCaseResultUpsert,
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.artifacts.settings_snapshot import write_settings_snapshot
from rulekiln.artifacts.writer import (
    write_baseline_prompt,
    write_cases_normalized,
    write_eval_report,
    write_manifest,
    write_mlflow_run_id,
    write_prompt,
    write_selected_prompt,
    write_strategy_comparison,
    write_task,
    write_token_cost_summary,
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
from rulekiln.pipeline.clustering import cluster_dbscan, cluster_hdbscan
from rulekiln.pipeline.evaluator import evaluate_prompt, get_primary_metric
from rulekiln.pipeline.failure_analysis import analyze_failures
from rulekiln.pipeline.prompt_compiler import (
    compile_baseline_prompt,
    compile_prompt,
    count_tokens_approx,
)
from rulekiln.pipeline.quality_gates import check_quality_gates
from rulekiln.pipeline.rule_pruning import prune_rules
from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.pipeline.strategy_selection import build_strategy_comparison
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.embedding import get_embedding_client
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.providers.tracking import (
    ModelCallCollector,
    ModelCallContext,
    set_tracking_context,
    update_tracking_context,
)
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    MicroRuleSchema,
    OutcomeCondition,
    QualityGateResult,
    RuleClusterSchema,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase
from rulekiln.workers.error_classification import format_worker_error_message

logger = get_logger(__name__)

_CASE_ID_DELIMITER = "::"


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
    run_dbscan_eval = phase in {"full", "evaluate_dbscan"}
    run_hdbscan_eval = phase in {"full", "evaluate_hdbscan"}
    run_aggregate = phase in {"full", "aggregate_evaluation_report"}

    teacher_profile = payload.teacher.provider_profile
    student_profile = payload.student.provider_profile
    embedding_profile = payload.embedding.provider_profile
    judge_route = payload.judge or payload.teacher
    judge_profile = judge_route.provider_profile

    # ── Stage: validating_project ──────────────────────────────────────────
    if run_validate and not await is_stage_complete(session, job_id, PipelineStage.VALIDATING_PROJECT):
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
                    provider_profile=teacher_profile,
                    provider=teacher_config.provider,
                    model=teacher_config.model,
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
                extraction = await extract_rules_for_case(task, case, teacher_chat, teacher_config)
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
        for strategy, clusters in [("dbscan", dbscan_clusters), ("hdbscan", hdbscan_clusters)]:
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
                    provider_profile=teacher_profile,
                    provider=teacher_config.provider,
                    model=teacher_config.model,
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
                case_ids = list({rule_map[rid].case_id for rid in cluster.rule_ids if rid in rule_map})
                case_ids = [_payload_case_id_from_db_case_id(job_id, case_id) for case_id in case_ids]
                synthesis = await synthesize_cluster(
                    task,
                    cluster.topic,
                    cluster_micro,
                    case_ids,
                    cluster.rule_ids,
                    teacher_chat,
                    teacher_config,
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

        # ── Stage: reviewing_rule_conflicts ──────────────────────────────
        total_train_cases = len(extraction_cases)
        for strategy in ("dbscan", "hdbscan"):
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
        for strategy in ("dbscan", "hdbscan"):
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
        for strategy in ("dbscan", "hdbscan"):
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

    baseline_eval: EvalResult | None = None
    if run_baseline_eval and not await is_stage_complete(session, job_id, PipelineStage.EVALUATING_BASELINE):
        await _set_stage(session, job_id, PipelineStage.EVALUATING_BASELINE)
        set_tracking_context(
            ModelCallContext(
                job_id=job_id,
                stage=PipelineStage.EVALUATING_BASELINE,
                role="student",
                provider_profile=student_profile,
                provider=student_config.provider,
                model=student_config.model,
                strategy="baseline",
            ),
            collector,
        )
        existing_baseline_rows = await get_eval_case_results(
            session,
            job_id=job_id,
            student_id=eval_student_id,
            strategy="baseline",
            split=eval_split,
        )
        completed_baseline_results = {
            row.case_id: _eval_case_record_to_schema(row)
            for row in existing_baseline_rows
        }

        async def _persist_baseline_case(case_result: CaseEvalResult) -> None:
            case = case_by_id.get(case_result.case_id)
            if case is None:
                return
            payload_row = _build_eval_case_upsert_payload(
                job_id=job_id,
                student_id=eval_student_id,
                strategy="baseline",
                split=eval_split,
                case=case,
                result=case_result,
            )
            await upsert_eval_case_result(session, payload_row)

        baseline_eval = await evaluate_prompt(
            baseline_prompt_text,
            eval_cases,
            task,
            student_chat,
            student_config,
            strategy="baseline",
            split=eval_split,
            completed_case_results=completed_baseline_results,
            on_case_result=_persist_baseline_case,
        )
        existing_eval_runs = await get_eval_runs_for_job(session, job_id)
        has_baseline_eval = any(
            run.strategy == "baseline" and run.split == eval_split for run in existing_eval_runs
        )
        if not has_baseline_eval:
            await insert_eval_run(session, _eval_to_db(job_id, None, baseline_eval))
        await mark_stage_complete(session, job_id, PipelineStage.EVALUATING_BASELINE)

    if phase == "evaluate_baseline":
        await _persist_collector_records(session, job_id, collector)
        return

    # ── Stage: evaluating_distilled ───────────────────────────────────────
    eval_map: dict[str, EvalResult] = {}
    distilled_strategies: tuple[str, ...]
    if phase == "evaluate_dbscan":
        distilled_strategies = ("dbscan",)
    elif phase == "evaluate_hdbscan":
        distilled_strategies = ("hdbscan",)
    elif phase == "full":
        distilled_strategies = ("dbscan", "hdbscan")
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
            row.case_id: _eval_case_record_to_schema(row)
            for row in existing_rows
        }

        async def _persist_strategy_case(case_result: CaseEvalResult) -> None:
            case = case_by_id.get(case_result.case_id)
            if case is None:
                return
            payload_row = _build_eval_case_upsert_payload(
                job_id=job_id,
                student_id=eval_student_id,
                strategy=strategy,
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
        if baseline_eval is None:
            baseline_eval = await _load_eval_result_from_db(
                session,
                job_id=job_id,
                student_id=eval_student_id,
                strategy="baseline",
                split=eval_split,
            )
        for strategy in ("dbscan", "hdbscan"):
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
        for strategy in ("dbscan", "hdbscan"):
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
            gate = check_quality_gates(
                strategy=strategy,
                distilled_eval=strategy_eval,
                baseline_eval=baseline_eval,
                cases=cases,
                task_mode=task.task_mode,
                task_gates=task.quality_gates,
                settings_defaults=settings.default_quality_gate,
                prompt_token_count=count_tokens_approx(prompt_text),
            )
            gate_map[strategy] = gate
            if not gate_stage_complete:
                await mark_stage_complete(
                    session, job_id, PipelineStage.CHECKING_QUALITY_GATES, strategy=strategy
                )

        token_counts: dict[str, int] = {}
        compiled_prompts: dict[str, str] = {}
        for strategy in ("dbscan", "hdbscan"):
            db_synth = await get_selected_synthesized_rules_for_job(session, job_id, strategy)
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            prompt_text, _ = compile_prompt(task, schemas, strategy)
            compiled_prompts[strategy] = prompt_text
            token_counts[strategy] = count_tokens_approx(prompt_text)

        comparison = build_strategy_comparison(
            baseline_eval=baseline_eval,
            dbscan_eval=eval_map.get("dbscan"),
            hdbscan_eval=eval_map.get("hdbscan"),
            dbscan_gate=gate_map.get("dbscan"),
            hdbscan_gate=gate_map.get("hdbscan"),
            task_mode=task.task_mode,
            dbscan_token_count=token_counts.get("dbscan", 0),
            hdbscan_token_count=token_counts.get("hdbscan", 0),
        )
        comparison.evaluation_split_warning = split_policy.fallback_warning
        selected_strategy = comparison.selected_strategy or "baseline"

        # ── Stage: selecting_strategy ─────────────────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.SELECTING_STRATEGY):
            await _set_stage(session, job_id, PipelineStage.SELECTING_STRATEGY)
            if selected_strategy != "baseline":
                await mark_prompt_version_selected(session, job_id, selected_strategy)
            logger.info(
                "strategy_selected",
                job_id=job_id,
                strategy=selected_strategy,
                reason=comparison.selection_reason,
            )
            await mark_stage_complete(session, job_id, PipelineStage.SELECTING_STRATEGY)

        # ── Stage: analyzing_failures ─────────────────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.ANALYZING_FAILURES):
            await _set_stage(session, job_id, PipelineStage.ANALYZING_FAILURES)
            selected_eval = eval_map.get(selected_strategy)
            if selected_eval and selected_strategy in {"dbscan", "hdbscan"}:
                db_synth = await get_selected_synthesized_rules_for_job(
                    session, job_id, selected_strategy
                )
                selected_schemas = [_db_synth_to_schema(r) for r in db_synth]
                analyze_failures(baseline_eval, selected_eval, selected_schemas)
            await mark_stage_complete(session, job_id, PipelineStage.ANALYZING_FAILURES)

        # ── Late stages ───────────────────────────────────────────────────
        if not await is_stage_complete(session, job_id, PipelineStage.LOGGING_ARTIFACTS):
            await _set_stage(session, job_id, PipelineStage.LOGGING_ARTIFACTS)
            selected_prompt_text = compiled_prompts.get(selected_strategy)
            if selected_strategy in {"dbscan", "hdbscan"} and selected_prompt_text is None:
                selected_db = await get_selected_synthesized_rules_for_job(
                    session, job_id, selected_strategy
                )
                selected_schemas = [_db_synth_to_schema(r) for r in selected_db]
                selected_prompt_text, _ = compile_prompt(task, selected_schemas, selected_strategy)
                compiled_prompts[selected_strategy] = selected_prompt_text

            prompt_hash_source = selected_prompt_text or baseline_prompt_text
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

            selected_eval_for_metrics = (
                baseline_eval
                if selected_strategy == "baseline"
                else eval_map.get(selected_strategy)
            )
            selected_gate = (
                gate_map.get(selected_strategy)
                if selected_strategy in {"dbscan", "hdbscan"}
                else None
            )
            eval_metrics = build_demo_eval_metrics(
                baseline_macro_f1=baseline_eval.macro_f1 if baseline_eval else None,
                baseline_accuracy=baseline_eval.accuracy if baseline_eval else None,
                baseline_malformed_output_rate=(
                    baseline_eval.malformed_output_rate if baseline_eval else None
                ),
                dbscan_macro_f1=eval_map.get("dbscan").macro_f1 if eval_map.get("dbscan") else None,
                dbscan_accuracy=eval_map.get("dbscan").accuracy if eval_map.get("dbscan") else None,
                dbscan_delta_vs_baseline=_delta_vs_baseline(
                    eval_map.get("dbscan"), baseline_eval, primary_metric
                ),
                hdbscan_macro_f1=(
                    eval_map.get("hdbscan").macro_f1 if eval_map.get("hdbscan") else None
                ),
                hdbscan_accuracy=(
                    eval_map.get("hdbscan").accuracy if eval_map.get("hdbscan") else None
                ),
                hdbscan_delta_vs_baseline=_delta_vs_baseline(
                    eval_map.get("hdbscan"), baseline_eval, primary_metric
                ),
                selected_primary_score=_primary_score_for_metric(
                    selected_eval_for_metrics,
                    primary_metric,
                ),
                selected_delta_vs_baseline=_delta_vs_baseline(
                    selected_eval_for_metrics,
                    baseline_eval,
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
                write_eval_report(artifact_root, comparison),
                write_strategy_comparison(artifact_root, comparison),
                write_settings_snapshot(artifact_root, settings),
            ]

            for strategy in ("dbscan", "hdbscan"):
                prompt_text = compiled_prompts.get(strategy)
                if prompt_text is None:
                    db_synth = await get_selected_synthesized_rules_for_job(session, job_id, strategy)
                    schemas = [_db_synth_to_schema(r) for r in db_synth]
                    prompt_text, _ = compile_prompt(task, schemas, strategy)
                    compiled_prompts[strategy] = prompt_text
                written_artifacts.append(write_prompt(artifact_root, strategy, prompt_text))

            if selected_strategy in {"dbscan", "hdbscan"}:
                selected_prompt_text = compiled_prompts.get(selected_strategy)
                if selected_prompt_text:
                    written_artifacts.append(write_selected_prompt(artifact_root, selected_prompt_text))

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
        str(expected): {
            str(actual): int(count)
            for actual, count in actuals.items()
        }
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
    digest = sha1(cluster_key.encode("utf-8")).hexdigest()[:16]
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
    expected_json: dict[str, object] | str | None
    if isinstance(case.expected, dict):
        expected_json = case.expected
    elif isinstance(case.expected, str):
        expected_json = case.expected
    else:
        expected_json = None

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


def _resolve_primary_metric(payload: DistillationRequest, task_mode: str) -> str:
    if payload.metric and payload.metric.strip():
        return payload.metric.strip()
    return get_primary_metric(task_mode)


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


def _build_manifest_entries(root: "Path", paths: list[object]) -> list[str]:
    from pathlib import Path

    entries: set[str] = set()
    for entry in paths:
        if not isinstance(entry, Path):
            continue
        if not entry.exists():
            continue
        try:
            entries.add(str(entry.relative_to(root)))
        except ValueError:
            continue
    return sorted(entries)


def _artifact_root(artifact_root_setting: str, job_id: str) -> "Path":
    from pathlib import Path

    return Path(artifact_root_setting) / job_id
