"""Stage services for the compile-phase tail: conflict review → pruning → prompt compilation.

StaticConflictReviewService  – run rule conflict review, update DB
PruningStageService          – select rules per token/support budget
PromptCompilationService     – compile final prompt, insert PromptVersion
"""

from __future__ import annotations

import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.agents.rule_conflict_review import review_rule_for_conflicts
from rulekiln.db.models import MicroRule, PromptVersion, SynthesizedRule
from rulekiln.db.repositories.jobs import (
    get_selected_synthesized_rules_for_job,
    get_synthesized_rules_for_job,
    insert_prompt_version,
    mark_stage_complete,
    update_synthesized_rule_pruning,
)
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.prompt_compiler import compile_prompt, count_tokens_approx
from rulekiln.pipeline.rule_pruning import prune_rules
from rulekiln.schemas.pipeline import MicroRuleSchema
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.db_mappers import _db_synth_to_schema
from rulekiln.workers.pipeline.paths import _conflict_review_rule_marker
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.stage_guard import StageGuard
from rulekiln.workers.pipeline.strategy_utils import DISTILLED_STRATEGIES

logger = get_logger(__name__)


# ── StaticConflictReviewService ─────────────────────────────────────────────


class StaticConflictReviewService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        rule_map: dict[str, MicroRule],
    ) -> None:
        from rulekiln.providers.tracking import (  # noqa: PLC0415
            ModelCallContext,
            set_tracking_context,
        )

        total_train_cases = len(ctx.extraction_cases)
        judge_profile = (ctx.payload.judge or ctx.payload.teacher).provider_profile
        guard = StageGuard(self._session, ctx.job_id)

        for strategy in DISTILLED_STRATEGIES:
            if await guard.is_complete(PipelineStage.REVIEWING_RULE_CONFLICTS, strategy=strategy):
                continue

            await guard.start(PipelineStage.REVIEWING_RULE_CONFLICTS)
            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.REVIEWING_RULE_CONFLICTS,
                    role="judge",
                    provider_profile=judge_profile,
                    provider=bundle.judge_config.provider,
                    model=bundle.judge_config.model,
                    strategy=strategy,
                ),
                ctx.collector,
            )

            db_synth = await get_synthesized_rules_for_job(self._session, ctx.job_id, strategy)
            reviewed_count = 0
            skipped_count = 0

            for synth_rule in db_synth:
                review_marker = _conflict_review_rule_marker(strategy, synth_rule.id)
                if await guard.is_complete(
                    PipelineStage.REVIEWING_RULE_CONFLICTS,
                    strategy=strategy,
                    artifact_type=review_marker,
                ):
                    skipped_count += 1
                    continue

                schema = _db_synth_to_schema(synth_rule)
                support = len(schema.source_case_ids)
                ratio = support / total_train_cases if total_train_cases > 0 else 0.0
                golden_backed = bool(
                    ctx.task.quality_gates and any(cid in set(schema.source_case_ids) for cid in [])
                )
                token_count = count_tokens_approx(f"{schema.topic} {' '.join(schema.applies_when)}")
                await update_synthesized_rule_pruning(
                    self._session,
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
                    ctx.task, schema, cluster_micro, bundle.judge_chat, bundle.judge_config
                )
                has_conflicts = review.has_conflicts and review.resolution in ("discard",)
                await self._session.execute(
                    update(SynthesizedRule)
                    .where(SynthesizedRule.id == synth_rule.id)
                    .values(
                        has_conflicts=has_conflicts,
                        conflict_summary=review.conflict_summary,
                        conflicting_micro_rule_ids=review.conflicting_micro_rule_ids,
                    )
                )
                await mark_stage_complete(
                    self._session,
                    ctx.job_id,
                    PipelineStage.REVIEWING_RULE_CONFLICTS,
                    strategy=strategy,
                    artifact_type=review_marker,
                )
                reviewed_count += 1

            await mark_stage_complete(
                self._session,
                ctx.job_id,
                PipelineStage.REVIEWING_RULE_CONFLICTS,
                strategy=strategy,
            )
            logger.info(
                "conflict_review_done",
                job_id=ctx.job_id,
                strategy=strategy,
                reviewed=reviewed_count,
                skipped=skipped_count,
            )


# ── PruningStageService ─────────────────────────────────────────────────────


class PruningStageService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(self, ctx: PipelineContext) -> None:
        guard = StageGuard(self._session, ctx.job_id)

        for strategy in DISTILLED_STRATEGIES:
            if await guard.is_complete(PipelineStage.PRUNING_RULES, strategy=strategy):
                continue

            await guard.start(PipelineStage.PRUNING_RULES)
            db_synth = await get_synthesized_rules_for_job(self._session, ctx.job_id, strategy)
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            pruning_result = prune_rules(
                schemas,
                max_rules=ctx.task.max_rules,
                max_prompt_tokens=ctx.task.max_prompt_tokens,
                min_rule_support_count=ctx.task.min_rule_support_count,
                preserve_golden_rules=ctx.task.preserve_golden_rules,
            )
            for record in pruning_result.pruned:
                await update_synthesized_rule_pruning(
                    self._session,
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
                job_id=ctx.job_id,
                strategy=strategy,
                selected=len(pruning_result.selected),
                pruned=len(pruning_result.pruned),
            )
            await mark_stage_complete(
                self._session, ctx.job_id, PipelineStage.PRUNING_RULES, strategy=strategy
            )


# ── PromptCompilationService ────────────────────────────────────────────────


class PromptCompilationService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(self, ctx: PipelineContext) -> None:
        guard = StageGuard(self._session, ctx.job_id)

        for strategy in DISTILLED_STRATEGIES:
            if await guard.is_complete(PipelineStage.COMPILING_PROMPTS, strategy=strategy):
                continue

            await guard.start(PipelineStage.COMPILING_PROMPTS)
            db_synth = await get_selected_synthesized_rules_for_job(
                self._session, ctx.job_id, strategy
            )
            schemas = [_db_synth_to_schema(r) for r in db_synth]
            prompt_text, prompt_hash = compile_prompt(ctx.task, schemas, strategy)
            pv = PromptVersion(
                id=str(uuid.uuid4()),
                job_id=ctx.job_id,
                task_id=ctx.task.task_id,
                task_name=ctx.task.task_name,
                strategy=strategy,
                version="v1",
                system_prompt=prompt_text,
                prompt_hash=prompt_hash,
            )
            await insert_prompt_version(self._session, pv)
            await mark_stage_complete(
                self._session, ctx.job_id, PipelineStage.COMPILING_PROMPTS, strategy=strategy
            )
