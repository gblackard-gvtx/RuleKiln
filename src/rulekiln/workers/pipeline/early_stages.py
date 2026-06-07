"""Stage services for the compile-phase pipeline: validation → synthesis.

ValidationStageService  – insert cases, VALIDATING_PROJECT marker
ExtractionStageService  – batch-submit, batch-collect, or sequential extraction
RuleEmbeddingStageService – embed micro-rule texts
ClusteringStageService  – run DBSCAN/HDBSCAN, persist clusters
SynthesisStageService   – synthesize per-cluster rules, persist
"""

from __future__ import annotations

import asyncio
import uuid
from typing import cast

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.agents.rule_extraction import (
    build_extraction_batch_item,
    extract_rules_for_case,
)
from rulekiln.agents.rule_synthesis import synthesize_cluster
from rulekiln.db.models import BatchJob as BatchJobModel
from rulekiln.db.models import MicroRule
from rulekiln.db.repositories.jobs import (
    bulk_insert_cases,
    bulk_insert_micro_rules,
    bulk_insert_rule_clusters,
    get_batch_job_by_stage,
    get_micro_rules_for_job,
    insert_batch_job,
    mark_stage_complete,
    update_batch_job,
)
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.clustering import cluster_dbscan, cluster_hdbscan
from rulekiln.providers.contracts import BatchChatModelClient
from rulekiln.providers.tracking import (
    ModelCallContext,
    set_tracking_context,
    update_tracking_context,
)
from rulekiln.schemas.classroom import PhaseTeacherConfig
from rulekiln.schemas.pipeline import ExtractionOutput, MicroRuleSchema, RuleClusterSchema
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.db_mappers import (
    _db_case_id,
    _payload_case_id_from_db_case_id,
    _synth_to_db,
    _to_db_case,
    _to_db_cluster,
)
from rulekiln.workers.pipeline.paths import (
    _extracting_case_marker,
    _synthesis_cluster_marker,
)
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.stage_guard import StageGuard
from rulekiln.workers.pipeline.strategy_utils import DISTILLED_STRATEGIES

logger = get_logger(__name__)


class RuleEmbeddingResult(BaseModel):
    """Output of RuleEmbeddingStageService, consumed by ClusteringStageService."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_micro_rules: list[MicroRule]
    rule_ids: list[str]
    rule_texts: list[str]
    embeddings: list[list[float]]


class ClusteringResult(BaseModel):
    """Output of ClusteringStageService, consumed by SynthesisStageService."""

    dbscan_clusters: list[RuleClusterSchema]
    hdbscan_clusters: list[RuleClusterSchema]


# ── ValidationStageService ──────────────────────────────────────────────────


class ValidationStageService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(self, ctx: PipelineContext) -> None:
        guard = StageGuard(self._session, ctx.job_id)

        async def _body() -> None:
            await bulk_insert_cases(self._session, [_to_db_case(ctx.job_id, c) for c in ctx.cases])

        await guard.run_once(PipelineStage.VALIDATING_PROJECT, fn=_body)


# ── ExtractionStageService ──────────────────────────────────────────────────


class ExtractionStageService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        *,
        phase: str,
    ) -> bool:
        """Run the extraction stage.

        Returns True when a batch job was submitted and the pipeline should halt
        (to be resumed via the extraction_batch_collect phase).  False means work
        is complete and the pipeline may continue.
        """
        extraction_phase_cfg: PhaseTeacherConfig | None = (
            ctx.payload.teacher_config.for_phase("instruction_extraction")
            if ctx.payload.teacher_config is not None
            else None
        )
        _extraction_profile = ctx.settings.provider_profiles.get(
            bundle.extraction_teacher_config.profile_name
        )
        use_batch = (
            extraction_phase_cfg is not None
            and extraction_phase_cfg.batch_enabled
            and isinstance(bundle.extraction_teacher_chat, BatchChatModelClient)
            and _extraction_profile is not None
            and _extraction_profile.batch_enabled
        )

        if use_batch:
            use_batch = await self._try_submit_batch(
                ctx,
                bundle,
                extraction_phase_cfg,  # type: ignore[arg-type]
            )

        if use_batch and phase == "extraction_batch_submit":
            return True

        if use_batch:
            await self._collect_batch(ctx, bundle)

        if not use_batch:
            await self._sequential(ctx, bundle)
        return False

    async def _try_submit_batch(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        extraction_phase_cfg: PhaseTeacherConfig,
    ) -> bool:
        """Submit a batch extraction job.

        Returns False if there are too few pending cases and the call site should
        fall back to sequential extraction; True in all other cases (already
        submitted or successfully submitted now).
        """
        from rulekiln.db.repositories.jobs import is_stage_complete  # noqa: PLC0415

        if await is_stage_complete(
            self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES_BATCH_SUBMITTED
        ):
            return True

        existing_micro_rules = await get_micro_rules_for_job(self._session, ctx.job_id)
        extracted_ids: set[str] = {
            _payload_case_id_from_db_case_id(ctx.job_id, r.case_id) for r in existing_micro_rules
        }
        pending_cases = [
            c
            for c in ctx.extraction_cases
            if c.id not in extracted_ids
            and not await is_stage_complete(
                self._session,
                ctx.job_id,
                PipelineStage.EXTRACTING_RULES,
                artifact_type=_extracting_case_marker(c.id),
            )
        ]

        min_items = extraction_phase_cfg.batch_min_items
        if len(pending_cases) < min_items:
            return False

        batch_items = [build_extraction_batch_item(ctx.task, c) for c in pending_cases]
        assert isinstance(bundle.extraction_teacher_chat, BatchChatModelClient)  # noqa: S101
        provider_batch_id = await bundle.extraction_teacher_chat.submit_batch(
            batch_items, bundle.extraction_teacher_config
        )
        db_batch_job = BatchJobModel(
            job_id=ctx.job_id,
            stage=PipelineStage.EXTRACTING_RULES,
            provider=bundle.extraction_teacher_config.provider,
            provider_batch_id=provider_batch_id,
            item_count=len(batch_items),
            metadata_json={"output_schema_class_name": "ExtractionOutput"},
        )
        await insert_batch_job(self._session, db_batch_job)
        await mark_stage_complete(
            self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES_BATCH_SUBMITTED
        )
        logger.info(
            "extraction_batch_submitted",
            job_id=ctx.job_id,
            provider_batch_id=provider_batch_id,
            item_count=len(batch_items),
        )
        return True

    async def _collect_batch(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
    ) -> None:
        from rulekiln.db.repositories.jobs import is_stage_complete  # noqa: PLC0415

        if await is_stage_complete(
            self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES_BATCH_COLLECTED
        ):
            return

        batch_job_rec = await get_batch_job_by_stage(
            self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES, strategy=None
        )
        if batch_job_rec is None:
            return

        assert isinstance(bundle.extraction_teacher_chat, BatchChatModelClient)  # noqa: S101
        poll_interval = ctx.settings.batch_poll_interval_seconds
        while True:
            poll_status = await bundle.extraction_teacher_chat.poll_batch(
                batch_job_rec.provider_batch_id, bundle.extraction_teacher_config
            )
            if not poll_status.processing:
                break
            await asyncio.sleep(poll_interval)

        schema_name: str = (batch_job_rec.metadata_json or {}).get(
            "output_schema_class_name", "ExtractionOutput"
        )  # type: ignore[assignment]
        batch_result = await bundle.extraction_teacher_chat.collect_batch(
            batch_job_rec.provider_batch_id,
            bundle.extraction_teacher_config,
            output_schema_class_name=schema_name,
        )

        inserted_count = 0
        failed_count = 0
        for item_result in batch_result.items:
            if (
                item_result.status == "succeeded"
                and item_result.result is not None
                and isinstance(item_result.result.parsed, ExtractionOutput)
            ):
                extraction = cast(ExtractionOutput, item_result.result.parsed)
                case_micro_rules: list[MicroRule] = [
                    MicroRule(
                        id=str(uuid.uuid4()),
                        job_id=ctx.job_id,
                        case_id=_db_case_id(ctx.job_id, item_result.custom_id),
                        topic=rule.topic,
                        condition=rule.condition,
                        expected_outcome=rule.expected_outcome,
                        output_path=rule.output_path,
                        rationale_summary=rule.rationale_summary,
                        rule_type=rule.rule_type,
                        positive_cues=rule.positive_cues,
                        negative_cues=rule.negative_cues,
                    )
                    for rule in extraction.rules  # type: ignore[union-attr]
                ]
                if case_micro_rules:
                    await bulk_insert_micro_rules(self._session, case_micro_rules)
                    inserted_count += len(case_micro_rules)
                await mark_stage_complete(
                    self._session,
                    ctx.job_id,
                    PipelineStage.EXTRACTING_RULES,
                    artifact_type=_extracting_case_marker(item_result.custom_id),
                )
            else:
                failed_count += 1
                logger.warning(
                    "extraction_batch_item_failed",
                    job_id=ctx.job_id,
                    custom_id=item_result.custom_id,
                    status=item_result.status,
                    error=item_result.error_message,
                )

        await update_batch_job(
            self._session,
            batch_job_rec.id,
            status=batch_result.status,
            succeeded_count=batch_result.succeeded_count,
            errored_count=batch_result.errored_count,
        )
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES)
        await mark_stage_complete(
            self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES_BATCH_COLLECTED
        )
        logger.info(
            "extraction_batch_collected",
            job_id=ctx.job_id,
            inserted_rules=inserted_count,
            failed_items=failed_count,
        )

    async def _sequential(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
    ) -> None:
        from rulekiln.db.repositories.jobs import (  # noqa: PLC0415
            is_stage_complete,
            update_job_status,
        )

        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES):
            return

        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.EXTRACTING_RULES
        )
        existing_micro_rules = await get_micro_rules_for_job(self._session, ctx.job_id)
        extracted_case_ids: set[str] = {
            _payload_case_id_from_db_case_id(ctx.job_id, db_rule.case_id)
            for db_rule in existing_micro_rules
        }
        set_tracking_context(
            ModelCallContext(
                job_id=ctx.job_id,
                stage=PipelineStage.EXTRACTING_RULES,
                role="teacher",
                provider_profile=bundle.extraction_teacher_config.profile_name,
                provider=bundle.extraction_teacher_config.provider,
                model=bundle.extraction_teacher_config.model,
            ),
            ctx.collector,
        )
        inserted_rule_count = 0
        skipped_case_count = 0
        for case in ctx.extraction_cases:
            case_marker = _extracting_case_marker(case.id)
            if case.id in extracted_case_ids or await is_stage_complete(
                self._session,
                ctx.job_id,
                PipelineStage.EXTRACTING_RULES,
                artifact_type=case_marker,
            ):
                skipped_case_count += 1
                continue

            update_tracking_context(case_id=case.id)
            extraction = await extract_rules_for_case(
                ctx.task, case, bundle.extraction_teacher_chat, bundle.extraction_teacher_config
            )
            new_rules: list[MicroRule] = []
            for rule in extraction.rules:  # type: ignore[union-attr]
                new_rules.append(
                    MicroRule(
                        id=str(uuid.uuid4()),
                        job_id=ctx.job_id,
                        case_id=_db_case_id(ctx.job_id, case.id),
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
            if new_rules:
                await bulk_insert_micro_rules(self._session, new_rules)
                inserted_rule_count += len(new_rules)

            await mark_stage_complete(
                self._session,
                ctx.job_id,
                PipelineStage.EXTRACTING_RULES,
                artifact_type=case_marker,
            )
            extracted_case_ids.add(case.id)

        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.EXTRACTING_RULES)
        logger.info(
            "rules_extracted",
            job_id=ctx.job_id,
            count=inserted_rule_count,
            skipped_cases=skipped_case_count,
        )


# ── RuleEmbeddingStageService ───────────────────────────────────────────────


class RuleEmbeddingStageService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
    ) -> RuleEmbeddingResult:
        db_micro_rules = await get_micro_rules_for_job(self._session, ctx.job_id)
        rule_ids = [r.id for r in db_micro_rules]
        rule_texts = [f"{r.topic}: {r.condition} → {r.expected_outcome}" for r in db_micro_rules]

        embeddings: list[list[float]] = []
        if rule_texts:
            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.EMBEDDING_RULES,
                    role="embedding",
                    provider_profile=ctx.payload.embedding.provider_profile,
                    provider=bundle.embedding_config.provider,
                    model=bundle.embedding_config.model,
                ),
                ctx.collector,
            )
            embedding_result = await bundle.embedding_client.embed_texts(
                texts=rule_texts, config=bundle.embedding_config
            )
            embeddings = embedding_result.embeddings

        guard = StageGuard(self._session, ctx.job_id)
        if not await guard.is_complete(PipelineStage.EMBEDDING_RULES):
            await guard.start(PipelineStage.EMBEDDING_RULES)
            await guard.complete(PipelineStage.EMBEDDING_RULES)

        return RuleEmbeddingResult(
            db_micro_rules=db_micro_rules,
            rule_ids=rule_ids,
            rule_texts=rule_texts,
            embeddings=embeddings,
        )


# ── ClusteringStageService ──────────────────────────────────────────────────


class ClusteringStageService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        embedding_result: RuleEmbeddingResult,
    ) -> ClusteringResult:
        rule_ids = embedding_result.rule_ids
        embeddings = embedding_result.embeddings

        dbscan_clusters = cluster_dbscan(rule_ids, embeddings) if rule_ids else []
        hdbscan_clusters = cluster_hdbscan(rule_ids, embeddings) if rule_ids else []

        guard = StageGuard(self._session, ctx.job_id)
        if not await guard.is_complete(PipelineStage.CLUSTERING_RULES):
            await guard.start(PipelineStage.CLUSTERING_RULES)
            db_clusters = [
                _to_db_cluster(ctx.job_id, c) for c in dbscan_clusters + hdbscan_clusters
            ]
            await bulk_insert_rule_clusters(self._session, db_clusters)
            await guard.complete(PipelineStage.CLUSTERING_RULES)

        return ClusteringResult(
            dbscan_clusters=dbscan_clusters,
            hdbscan_clusters=hdbscan_clusters,
        )


# ── SynthesisStageService ───────────────────────────────────────────────────


class SynthesisStageService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        clustering_result: ClusteringResult,
        rule_map: dict[str, MicroRule],
    ) -> None:
        from rulekiln.db.repositories.jobs import (  # noqa: PLC0415
            is_stage_complete,
            update_job_status,
        )

        for strategy, clusters in [
            (DISTILLED_STRATEGIES[0], clustering_result.dbscan_clusters),
            (DISTILLED_STRATEGIES[1], clustering_result.hdbscan_clusters),
        ]:
            if await is_stage_complete(
                self._session, ctx.job_id, PipelineStage.SYNTHESIZING_RULES, strategy=strategy
            ):
                continue

            await update_job_status(
                self._session,
                ctx.job_id,
                status="running",
                stage=PipelineStage.SYNTHESIZING_RULES,
            )
            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.SYNTHESIZING_RULES,
                    role="teacher",
                    provider_profile=bundle.synthesis_teacher_config.profile_name,
                    provider=bundle.synthesis_teacher_config.provider,
                    model=bundle.synthesis_teacher_config.model,
                    strategy=strategy,
                ),
                ctx.collector,
            )

            inserted_rule_count = 0
            processed_cluster_count = 0
            skipped_cluster_count = 0

            for cluster in clusters:
                cluster_marker = _synthesis_cluster_marker(strategy, cluster.rule_ids)
                if await is_stage_complete(
                    self._session,
                    ctx.job_id,
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
                case_ids = [_payload_case_id_from_db_case_id(ctx.job_id, cid) for cid in case_ids]
                synthesis = await synthesize_cluster(
                    ctx.task,
                    cluster.topic,
                    cluster_micro,
                    case_ids,
                    cluster.rule_ids,
                    bundle.synthesis_teacher_chat,
                    bundle.synthesis_teacher_config,
                )
                cluster_synth_rules = [
                    _synth_to_db(ctx.job_id, strategy, r) for r in synthesis.rules
                ]

                if cluster_synth_rules:
                    self._session.add_all(cluster_synth_rules)
                    inserted_rule_count += len(cluster_synth_rules)

                await mark_stage_complete(
                    self._session,
                    ctx.job_id,
                    PipelineStage.SYNTHESIZING_RULES,
                    strategy=strategy,
                    artifact_type=cluster_marker,
                )
                processed_cluster_count += 1

            await mark_stage_complete(
                self._session, ctx.job_id, PipelineStage.SYNTHESIZING_RULES, strategy=strategy
            )
            logger.info(
                "synthesis_done",
                job_id=ctx.job_id,
                strategy=strategy,
                processed_clusters=processed_cluster_count,
                skipped_clusters=skipped_cluster_count,
                inserted_rules=inserted_rule_count,
            )
