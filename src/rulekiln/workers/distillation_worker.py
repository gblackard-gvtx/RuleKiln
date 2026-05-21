"""Full distillation pipeline worker with stage orchestration, resume semantics, and idempotency."""

from __future__ import annotations

import uuid
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.agents.rule_extraction import extract_rules_for_case
from rulekiln.agents.rule_synthesis import synthesize_cluster
from rulekiln.config.settings import get_settings
from rulekiln.db.models import (
    Case,
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
    bulk_insert_synthesized_rules,
    get_micro_rules_for_job,
    get_synthesized_rules_for_job,
    insert_eval_run,
    insert_prompt_version,
    is_stage_complete,
    mark_prompt_version_selected,
    mark_stage_complete,
    update_job_status,
)
from rulekiln.db.session import get_session_factory
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.clustering import cluster_dbscan, cluster_hdbscan
from rulekiln.pipeline.evaluator import evaluate_prompt
from rulekiln.pipeline.failure_analysis import analyze_failures
from rulekiln.pipeline.prompt_compiler import compile_prompt, count_tokens_approx
from rulekiln.pipeline.quality_gates import check_quality_gates
from rulekiln.pipeline.strategy_selection import build_strategy_comparison
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.embedding import get_embedding_client
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import (
    EvalResult,
    MicroRuleSchema,
    OutcomeCondition,
    QualityGateResult,
    RuleClusterSchema,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase

logger = get_logger(__name__)


class PipelineStage(StrEnum):
    CREATED = "created"
    VALIDATING_PROJECT = "validating_project"
    EXTRACTING_RULES = "extracting_rules"
    EMBEDDING_RULES = "embedding_rules"
    CLUSTERING_RULES = "clustering_rules"
    SYNTHESIZING_RULES = "synthesizing_rules"
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
            logger.error("pipeline_failed", job_id=job_id, error=str(exc))
            await update_job_status(
                session, job_id, status="failed",
                stage=PipelineStage.FAILED, error_message=str(exc),
            )
            raise


async def _run(session: AsyncSession, job_id: str, payload: DistillationRequest) -> None:  # noqa: C901
    settings = get_settings()
    task = payload.task
    cases = payload.cases

    # ── Stage: validating_project ──────────────────────────────────────────
    if not await is_stage_complete(session, job_id, PipelineStage.VALIDATING_PROJECT):
        await _set_stage(session, job_id, PipelineStage.VALIDATING_PROJECT)
        await bulk_insert_cases(session, [_to_db_case(job_id, c) for c in cases])
        await mark_stage_complete(session, job_id, PipelineStage.VALIDATING_PROJECT)

    teacher_config = resolve_provider_config(
        payload.teacher.provider_profile, payload.teacher.model, role="teacher", settings=settings,
    )
    student_config = resolve_provider_config(
        payload.student.provider_profile, payload.student.model, role="student", settings=settings,
    )
    embedding_config = resolve_provider_config(
        payload.embedding.provider_profile, payload.embedding.model, role="embedding", settings=settings,
    )
    teacher_chat = get_chat_client(teacher_config)
    student_chat = get_chat_client(student_config)
    embedding_client = get_embedding_client(embedding_config)

    # ── Stage: extracting_rules ───────────────────────────────────────────
    if not await is_stage_complete(session, job_id, PipelineStage.EXTRACTING_RULES):
        await _set_stage(session, job_id, PipelineStage.EXTRACTING_RULES)
        train_cases = [c for c in cases if c.split in ("train", "validation")]
        micro_rules: list[MicroRule] = []
        for case in train_cases:
            extraction = await extract_rules_for_case(task, case, teacher_chat, teacher_config)
            for rule in extraction.rules:
                micro_rules.append(MicroRule(
                    id=str(uuid.uuid4()), job_id=job_id, case_id=case.id,
                    topic=rule.topic, condition=rule.condition,
                    expected_outcome=rule.expected_outcome, output_path=rule.output_path,
                    rationale_summary=rule.rationale_summary, rule_type=rule.rule_type,
                    positive_cues=rule.positive_cues, negative_cues=rule.negative_cues,
                ))
        await bulk_insert_micro_rules(session, micro_rules)
        await mark_stage_complete(session, job_id, PipelineStage.EXTRACTING_RULES)
        logger.info("rules_extracted", job_id=job_id, count=len(micro_rules))

    db_micro_rules = await get_micro_rules_for_job(session, job_id)
    rule_ids = [r.id for r in db_micro_rules]
    rule_texts = [f"{r.topic}: {r.condition} → {r.expected_outcome}" for r in db_micro_rules]

    # ── Stage: embedding_rules ────────────────────────────────────────────
    embeddings: list[list[float]] = []
    if rule_texts:
        embeddings = await embedding_client.embed_texts(texts=rule_texts, config=embedding_config)
    if not await is_stage_complete(session, job_id, PipelineStage.EMBEDDING_RULES):
        await _set_stage(session, job_id, PipelineStage.EMBEDDING_RULES)
        await mark_stage_complete(session, job_id, PipelineStage.EMBEDDING_RULES)

    # ── Stage: clustering_rules ───────────────────────────────────────────
    dbscan_clusters = cluster_dbscan(rule_ids, embeddings) if rule_ids else []
    hdbscan_clusters = cluster_hdbscan(rule_ids, embeddings) if rule_ids else []
    if not await is_stage_complete(session, job_id, PipelineStage.CLUSTERING_RULES):
        await _set_stage(session, job_id, PipelineStage.CLUSTERING_RULES)
        db_clusters = [_to_db_cluster(job_id, c) for c in dbscan_clusters + hdbscan_clusters]
        await bulk_insert_rule_clusters(session, db_clusters)
        await mark_stage_complete(session, job_id, PipelineStage.CLUSTERING_RULES)

    # ── Stage: synthesizing_rules ─────────────────────────────────────────
    rule_map: dict[str, MicroRule] = {r.id: r for r in db_micro_rules}
    for strategy, clusters in [("dbscan", dbscan_clusters), ("hdbscan", hdbscan_clusters)]:
        if await is_stage_complete(session, job_id, PipelineStage.SYNTHESIZING_RULES, strategy=strategy):
            continue
        await _set_stage(session, job_id, PipelineStage.SYNTHESIZING_RULES)
        synth_rules: list[SynthesizedRule] = []
        for cluster in clusters:
            cluster_micro = [
                MicroRuleSchema(
                    topic=rule_map[rid].topic, condition=rule_map[rid].condition,
                    expected_outcome=rule_map[rid].expected_outcome, output_path=rule_map[rid].output_path,
                    rationale_summary=rule_map[rid].rationale_summary, rule_type=rule_map[rid].rule_type,
                    positive_cues=list(rule_map[rid].positive_cues or []),
                    negative_cues=list(rule_map[rid].negative_cues or []),
                )
                for rid in cluster.rule_ids if rid in rule_map
            ]
            case_ids = list({rule_map[rid].case_id for rid in cluster.rule_ids if rid in rule_map})
            synthesis = await synthesize_cluster(
                task, cluster.topic, cluster_micro, case_ids, cluster.rule_ids,
                teacher_chat, teacher_config,
            )
            for rule in synthesis.rules:
                synth_rules.append(_synth_to_db(job_id, strategy, rule))
        await bulk_insert_synthesized_rules(session, synth_rules)
        await mark_stage_complete(session, job_id, PipelineStage.SYNTHESIZING_RULES, strategy=strategy)

    # ── Stage: compiling_prompts ──────────────────────────────────────────
    for strategy in ("dbscan", "hdbscan"):
        if await is_stage_complete(session, job_id, PipelineStage.COMPILING_PROMPTS, strategy=strategy):
            continue
        await _set_stage(session, job_id, PipelineStage.COMPILING_PROMPTS)
        db_synth = await get_synthesized_rules_for_job(session, job_id, strategy)
        schemas = [_db_synth_to_schema(r) for r in db_synth]
        prompt_text, prompt_hash = compile_prompt(task, schemas, strategy)
        pv = PromptVersion(
            id=str(uuid.uuid4()), job_id=job_id, task_id=task.task_id,
            task_name=task.task_name, strategy=strategy, version="v1",
            system_prompt=prompt_text, prompt_hash=prompt_hash,
        )
        await insert_prompt_version(session, pv)
        await mark_stage_complete(session, job_id, PipelineStage.COMPILING_PROMPTS, strategy=strategy)

    # ── Stage: evaluating_baseline ────────────────────────────────────────
    baseline_eval: EvalResult | None = None
    if not await is_stage_complete(session, job_id, PipelineStage.EVALUATING_BASELINE):
        await _set_stage(session, job_id, PipelineStage.EVALUATING_BASELINE)
        if payload.baseline_prompt:
            baseline_eval = await evaluate_prompt(
                payload.baseline_prompt, cases, task, student_chat, student_config, strategy="baseline",
            )
            await insert_eval_run(session, _eval_to_db(job_id, None, baseline_eval))
        await mark_stage_complete(session, job_id, PipelineStage.EVALUATING_BASELINE)

    # ── Stage: evaluating_distilled ───────────────────────────────────────
    eval_map: dict[str, EvalResult] = {}
    for strategy in ("dbscan", "hdbscan"):
        if await is_stage_complete(session, job_id, PipelineStage.EVALUATING_DISTILLED, strategy=strategy):
            continue
        await _set_stage(session, job_id, PipelineStage.EVALUATING_DISTILLED)
        db_synth = await get_synthesized_rules_for_job(session, job_id, strategy)
        schemas = [_db_synth_to_schema(r) for r in db_synth]
        prompt_text, _ = compile_prompt(task, schemas, strategy)
        ev = await evaluate_prompt(
            prompt_text, cases, task, student_chat, student_config, strategy=strategy,
        )
        eval_map[strategy] = ev
        await insert_eval_run(session, _eval_to_db(job_id, None, ev))
        await mark_stage_complete(session, job_id, PipelineStage.EVALUATING_DISTILLED, strategy=strategy)

    # ── Stage: checking_quality_gates ─────────────────────────────────────
    gate_map: dict[str, QualityGateResult] = {}
    for strategy in ("dbscan", "hdbscan"):
        if strategy not in eval_map:
            continue
        if await is_stage_complete(session, job_id, PipelineStage.CHECKING_QUALITY_GATES, strategy=strategy):
            continue
        await _set_stage(session, job_id, PipelineStage.CHECKING_QUALITY_GATES)
        db_synth = await get_synthesized_rules_for_job(session, job_id, strategy)
        schemas = [_db_synth_to_schema(r) for r in db_synth]
        prompt_text, _ = compile_prompt(task, schemas, strategy)
        gate = check_quality_gates(
            strategy=strategy, distilled_eval=eval_map[strategy], baseline_eval=baseline_eval,
            cases=cases, task_mode=task.task_mode, task_gates=task.quality_gates,
            settings_defaults=settings.default_quality_gate,
            prompt_token_count=count_tokens_approx(prompt_text),
        )
        gate_map[strategy] = gate
        await mark_stage_complete(session, job_id, PipelineStage.CHECKING_QUALITY_GATES, strategy=strategy)

    # ── Stage: selecting_strategy ─────────────────────────────────────────
    if not await is_stage_complete(session, job_id, PipelineStage.SELECTING_STRATEGY):
        await _set_stage(session, job_id, PipelineStage.SELECTING_STRATEGY)
        token_counts: dict[str, int] = {}
        for strategy in ("dbscan", "hdbscan"):
            db_synth = await get_synthesized_rules_for_job(session, job_id, strategy)
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            pt, _ = compile_prompt(task, schemas, strategy)
            token_counts[strategy] = count_tokens_approx(pt)
        comparison = build_strategy_comparison(
            baseline_eval=baseline_eval,
            dbscan_eval=eval_map.get("dbscan"), hdbscan_eval=eval_map.get("hdbscan"),
            dbscan_gate=gate_map.get("dbscan"), hdbscan_gate=gate_map.get("hdbscan"),
            task_mode=task.task_mode,
            dbscan_token_count=token_counts.get("dbscan", 0),
            hdbscan_token_count=token_counts.get("hdbscan", 0),
        )
        if comparison.selected_strategy and comparison.selected_strategy != "baseline":
            await mark_prompt_version_selected(session, job_id, comparison.selected_strategy)
        logger.info(
            "strategy_selected", job_id=job_id,
            strategy=comparison.selected_strategy, reason=comparison.selection_reason,
        )
        await mark_stage_complete(session, job_id, PipelineStage.SELECTING_STRATEGY)

    # ── Stage: analyzing_failures ─────────────────────────────────────────
    if not await is_stage_complete(session, job_id, PipelineStage.ANALYZING_FAILURES):
        await _set_stage(session, job_id, PipelineStage.ANALYZING_FAILURES)
        selected_eval = eval_map.get("hdbscan") or eval_map.get("dbscan")
        if selected_eval:
            analyze_failures(baseline_eval, selected_eval)
        await mark_stage_complete(session, job_id, PipelineStage.ANALYZING_FAILURES)

    # ── Late stages ───────────────────────────────────────────────────────
    for late in (PipelineStage.LOGGING_ARTIFACTS, PipelineStage.EXPORTING_ARTIFACTS):
        if not await is_stage_complete(session, job_id, late):
            await _set_stage(session, job_id, late)
            await mark_stage_complete(session, job_id, late)

    await update_job_status(session, job_id, status="completed", stage=PipelineStage.COMPLETED)
    logger.info("pipeline_completed", job_id=job_id)


# ── Helpers ──────────────────────────────────────────────────────────────────

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
        id=case.id, job_id=job_id, task_mode=case.task_mode, split=case.split,
        input_json=case.input, expected_json=expected_json, expected_text=expected_text,
        evaluation_json=case.evaluation.model_dump(), metadata_json=case.metadata, weight=case.weight,
    )


def _to_db_cluster(job_id: str, c: RuleClusterSchema) -> RuleCluster:
    return RuleCluster(
        id=str(uuid.uuid4()), job_id=job_id, strategy=c.strategy, topic=c.topic,
        algorithm=c.algorithm, rule_ids=c.rule_ids, cluster_metadata=c.cluster_metadata,
    )


def _synth_to_db(job_id: str, strategy: str, rule: SynthesizedRuleSchema) -> SynthesizedRule:
    return SynthesizedRule(
        id=str(uuid.uuid4()), job_id=job_id, strategy=strategy, topic=rule.topic,
        applies_when=rule.applies_when,
        outcome_conditions={k: v.model_dump() for k, v in rule.outcome_conditions.items()},
        tie_breakers=rule.tie_breakers, priority=rule.priority,
        source_case_ids=rule.source_case_ids, source_micro_rule_ids=rule.source_micro_rule_ids,
    )


def _db_synth_to_schema(r: SynthesizedRule) -> SynthesizedRuleSchema:
    oc = {k: OutcomeCondition.model_validate(v) for k, v in (r.outcome_conditions or {}).items()}
    return SynthesizedRuleSchema(
        topic=r.topic, applies_when=list(r.applies_when or []), outcome_conditions=oc,
        tie_breakers=list(r.tie_breakers or []), priority=r.priority,
        source_case_ids=list(r.source_case_ids or []),
        source_micro_rule_ids=list(r.source_micro_rule_ids or []),
    )


def _eval_to_db(job_id: str, prompt_version_id: str | None, result: EvalResult) -> EvalRun:
    return EvalRun(
        id=str(uuid.uuid4()), job_id=job_id, prompt_version_id=prompt_version_id,
        strategy=result.strategy, model=result.model, split=result.split,
        accuracy=result.accuracy, macro_f1=result.macro_f1,
        weighted_case_score=result.weighted_case_score,
        per_outcome_precision=result.per_outcome_precision,
        per_outcome_recall=result.per_outcome_recall,
        malformed_output_rate=result.malformed_output_rate,
        confusion_matrix={k: dict(v) for k, v in result.confusion_matrix.items()},
    )
