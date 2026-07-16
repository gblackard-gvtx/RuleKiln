"""Output pipeline services: classroom evaluation, MLflow logging, artifact export.

ClassroomEvaluationService  – evaluate all LLM strategies across classroom students
MlflowLoggingService        – create MLflow run, log params and metrics
ArtifactExportService       – write all job artifacts to disk, finalize manifest
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.artifacts.manifest import build_manifest_entries
from rulekiln.artifacts.settings_snapshot import write_settings_snapshot
from rulekiln.artifacts.writer import (
    write_artifact_hashes,
    write_baseline_prompt,
    write_baseline_scaffold_prompt,
    write_cases_normalized,
    write_confusion_matrix_csv,
    write_eval_report,
    write_integrity_check_failures,
    write_manifest,
    write_mlflow_run_id,
    write_paired_comparison_artifacts,
    write_per_label_metrics_csv,
    write_promotion_status,
    write_prompt,
    write_provenance_validation,
    write_rule_ablation_json,
    write_rule_provenance_json,
    write_rule_provenance_markdown,
    write_selected_prompt,
    write_strategy_comparison,
    write_strategy_debug_artifacts,
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
from rulekiln.db.repositories.jobs import (
    get_rule_clusters_for_job,
    get_selected_synthesized_rules_for_job,
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
from rulekiln.pipeline.approved_configs import load_approved_config
from rulekiln.pipeline.artifact_hashing import (
    build_artifact_hashes,
    build_job_hashes,
    build_strategy_provenance_row,
    clustering_config_hash,
)
from rulekiln.pipeline.baseline_strategies import (
    BASELINE_FEW_SHOT_STRATEGY_TO_K,
    BASELINE_SCAFFOLD_STRATEGY,
    build_few_shot_prompt_with_budget,
    select_deterministic_few_shot_examples,
)
from rulekiln.pipeline.classroom_evaluator import evaluate_classroom as _evaluate_classroom
from rulekiln.pipeline.failure_analysis import analyze_failures
from rulekiln.pipeline.promotion import evaluate_promotion
from rulekiln.pipeline.prompt_compiler import compile_prompt
from rulekiln.pipeline.provenance_validation import (
    load_clustering_provenance,
    validate_provenance,
)
from rulekiln.pipeline.rule_provenance import (
    build_cluster_id_by_micro_rule,
    build_provenance_records,
)
from rulekiln.pipeline.split_leakage_validation import (
    validate_split_leakage,
    write_split_leakage_validation,
)
from rulekiln.pipeline.stale_artifact_validation import (
    validate_stale_artifacts,
    write_stale_artifact_validation,
)
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.schemas.classroom import StudentConfig
from rulekiln.schemas.integrity_check_failure import (
    IntegrityCheckFailure,
    IntegrityCheckFailuresReport,
)
from rulekiln.schemas.pipeline import (
    RuleAblationArtifact,
    StrategyComparison,
)
from rulekiln.schemas.promotion import (
    PromotionPolicy,
    PromotionStatus,
    ProvenanceValidationReport,
)
from rulekiln.schemas.split_leakage import SplitLeakageValidationReport
from rulekiln.schemas.stale_artifact import StaleArtifactValidationReport
from rulekiln.workers.pipeline.artifacts_state import PipelineArtifactsState
from rulekiln.workers.pipeline.classification_metrics import (
    _resolve_training_cases,
)
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.db_mappers import _db_synth_to_schema
from rulekiln.workers.pipeline.paths import (
    _artifact_root,
    _build_manifest_entries,
)
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.seeding import _evaluation_bootstrap_seed
from rulekiln.workers.pipeline.strategy_utils import (
    DISTILLED_STRATEGIES,
    EMBEDDING_BASELINE_STRATEGIES,
    _delta_vs_baseline,
    _primary_score_for_metric,
)

logger = get_logger(__name__)

_BASELINE_FEW_SHOT_STRATEGIES: tuple[str, ...] = tuple(BASELINE_FEW_SHOT_STRATEGY_TO_K.keys())


class ClassroomEvaluationService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        state: PipelineArtifactsState,
        comparison: StrategyComparison,
    ) -> None:
        _classroom_cfg = ctx.payload.classroom_config
        if (
            _classroom_cfg is None
            or len(_classroom_cfg.students) <= 1
            or await is_stage_complete(
                self._session, ctx.job_id, PipelineStage.EVALUATING_CLASSROOM
            )
        ):
            return
        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.EVALUATING_CLASSROOM
        )
        logger.info("stage_started", job_id=ctx.job_id, stage=PipelineStage.EVALUATING_CLASSROOM)

        def _classroom_chat_client(student: StudentConfig) -> ChatModelClient:
            cfg = resolve_provider_config(
                student.provider, student.model, role="student", settings=ctx.settings
            )
            return get_chat_client(cfg)

        def _classroom_provider_config(student: StudentConfig) -> ProviderConfig:
            return resolve_provider_config(
                student.provider, student.model, role="student", settings=ctx.settings
            )

        _llm_prompt_strategies = {
            s: p
            for s, p in state.compiled_prompts.items()
            if s not in EMBEDDING_BASELINE_STRATEGIES
        }
        _all_student_results = {}
        for _cls_strategy, _cls_prompt in _llm_prompt_strategies.items():
            try:
                _cls_results = await _evaluate_classroom(
                    system_prompt=_cls_prompt,
                    cases=ctx.eval_cases,
                    task=ctx.task,
                    classroom_config=_classroom_cfg,
                    get_chat_client=_classroom_chat_client,
                    get_provider_config=_classroom_provider_config,
                    strategy=_cls_strategy,
                    split=ctx.eval_split,
                    bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                    bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                    bootstrap_seed=_evaluation_bootstrap_seed(
                        job_id=ctx.job_id,
                        strategy=_cls_strategy,
                        split=ctx.eval_split,
                        seed_offset=ctx.evaluation_bootstrap_seed_offset,
                    ),
                )
                state.classroom_student_evals[_cls_strategy] = _cls_results
                _all_student_results[_cls_strategy] = {
                    sid: student_eval_summary_from_eval_result(r, sid)
                    for sid, r in _cls_results.items()
                }
            except Exception as _cls_exc:
                logger.warning(
                    "classroom_eval_strategy_failed",
                    job_id=ctx.job_id,
                    strategy=_cls_strategy,
                    error=str(_cls_exc),
                )
        if _all_student_results:
            comparison.all_student_results = _all_student_results
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.EVALUATING_CLASSROOM)


class MlflowLoggingService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        bundle: ProviderBundle,
        state: PipelineArtifactsState,
        comparison: StrategyComparison,
        selected_strategy: str,
    ) -> None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.LOGGING_ARTIFACTS):
            return
        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.LOGGING_ARTIFACTS
        )
        logger.info("stage_started", job_id=ctx.job_id, stage=PipelineStage.LOGGING_ARTIFACTS)

        selected_prompt_text = state.compiled_prompts.get(selected_strategy)
        if selected_strategy in DISTILLED_STRATEGIES and selected_prompt_text is None:
            selected_db = await get_selected_synthesized_rules_for_job(
                self._session, ctx.job_id, selected_strategy
            )
            selected_schemas = [_db_synth_to_schema(r) for r in selected_db]
            selected_prompt_text, _ = compile_prompt(ctx.task, selected_schemas, selected_strategy)
            state.compiled_prompts[selected_strategy] = selected_prompt_text

        prompt_hash_source = selected_prompt_text or state.baseline_variant_prompts.get(
            BASELINE_SCAFFOLD_STRATEGY,
            state.baseline_prompt_text,
        )
        prompt_hash = sha256(prompt_hash_source.encode("utf-8")).hexdigest()

        mlflow_run_id = create_run(
            ctx.settings.mlflow_tracking_uri,
            ctx.settings.mlflow_experiment_name,
            job_id=ctx.job_id,
            task_id=ctx.task.task_id,
            task_name=ctx.task.task_name,
        )
        await set_mlflow_run_id(self._session, ctx.job_id, mlflow_run_id)
        artifact_root = _artifact_root(ctx.settings.artifact_root, ctx.job_id)
        write_mlflow_run_id(artifact_root, mlflow_run_id)

        run_params: dict[str, str] = {
            **build_run_params(
                job_id=ctx.job_id,
                task_id=ctx.task.task_id,
                strategy=selected_strategy,
                prompt_hash=prompt_hash,
            ),
            **build_provider_params(ctx.payload),
            **build_demo_params(
                task_id=ctx.task.task_id,
                task_mode=ctx.task.task_mode,
                dataset=ctx.dataset_id,
                teacher_provider=bundle.teacher_config.provider,
                teacher_model=bundle.teacher_config.model,
                student_provider=bundle.student_config.provider,
                student_model=bundle.student_config.model,
                embedding_model=bundle.embedding_config.model,
                selected_strategy=selected_strategy,
                primary_metric=ctx.primary_metric,
            ),
            "baseline_prompt_source": state.baseline_prompt_source,
            "baseline_prompt_compiler": ctx.task.baseline_prompt_policy.compiler,
        }
        log_params(ctx.settings.mlflow_tracking_uri, mlflow_run_id, run_params)

        eval_map = state.eval_map
        selected_eval_for_metrics = eval_map.get(selected_strategy)
        if selected_eval_for_metrics is None and selected_strategy == "baseline":
            selected_eval_for_metrics = state.baseline_eval
        selected_gate = state.gate_map.get(selected_strategy)
        dbscan_eval = eval_map.get("dbscan")
        hdbscan_eval = eval_map.get("hdbscan")
        baseline_reference_eval = state.baseline_scaffold_eval or state.baseline_eval
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
                ctx.primary_metric,
            ),
            hdbscan_macro_f1=hdbscan_eval.macro_f1 if hdbscan_eval else None,
            hdbscan_accuracy=hdbscan_eval.accuracy if hdbscan_eval else None,
            hdbscan_delta_vs_baseline=_delta_vs_baseline(
                hdbscan_eval,
                baseline_reference_eval,
                ctx.primary_metric,
            ),
            selected_primary_score=_primary_score_for_metric(
                selected_eval_for_metrics,
                ctx.primary_metric,
            ),
            selected_delta_vs_baseline=_delta_vs_baseline(
                selected_eval_for_metrics,
                baseline_reference_eval,
                ctx.primary_metric,
            ),
            selected_passed_quality_gates=(
                selected_gate.passed if selected_gate is not None else False
            ),
        )
        log_metrics(ctx.settings.mlflow_tracking_uri, mlflow_run_id, eval_metrics)

        logger.info(
            "mlflow_params_logged",
            job_id=ctx.job_id,
            run_id=mlflow_run_id,
            strategy=selected_strategy,
        )
        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.LOGGING_ARTIFACTS)


class ArtifactExportService:
    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        ctx: PipelineContext,
        state: PipelineArtifactsState,
        comparison: StrategyComparison,
        selected_strategy: str,
        ablation_artifact: RuleAblationArtifact | None,
    ) -> None:
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.EXPORTING_ARTIFACTS):
            return
        await update_job_status(
            self._session, ctx.job_id, status="running", stage=PipelineStage.EXPORTING_ARTIFACTS
        )
        logger.info("stage_started", job_id=ctx.job_id, stage=PipelineStage.EXPORTING_ARTIFACTS)

        artifact_root = _artifact_root(ctx.settings.artifact_root, ctx.job_id)
        written_artifacts = [
            write_task(artifact_root, ctx.task),
            write_cases_normalized(artifact_root, ctx.cases),
            write_baseline_prompt(artifact_root, state.baseline_prompt_text),
            write_baseline_scaffold_prompt(artifact_root, state.baseline_prompt_text),
            write_eval_report(artifact_root, comparison),
            write_strategy_comparison(artifact_root, comparison),
            write_settings_snapshot(artifact_root, ctx.settings),
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

        if state.paired_comparison_artifacts is not None:
            written_artifacts.extend(
                write_paired_comparison_artifacts(
                    artifact_root / "outputs" / "paired_comparison",
                    state.paired_comparison_artifacts,
                )
            )

        if ablation_artifact is not None:
            written_artifacts.append(write_rule_ablation_json(artifact_root, ablation_artifact))

        # Rule provenance
        if selected_strategy in DISTILLED_STRATEGIES:
            prov_selected_eval = comparison.strategy_evals.get(selected_strategy)
            prov_db_synth = await get_selected_synthesized_rules_for_job(
                self._session, ctx.job_id, selected_strategy
            )
            prov_schemas = [_db_synth_to_schema(r) for r in prov_db_synth]
            if prov_schemas:
                prov_failure_analysis = (
                    analyze_failures(
                        state.baseline_scaffold_eval or state.baseline_eval,
                        prov_selected_eval,
                        prov_schemas,
                    )
                    if prov_selected_eval
                    else None
                )
                db_clusters = await get_rule_clusters_for_job(
                    self._session, ctx.job_id, selected_strategy
                )
                cluster_pairs = [(c.id, list(c.rule_ids)) for c in db_clusters]
                cluster_id_map = build_cluster_id_by_micro_rule(cluster_pairs)
                prov_artifact = build_provenance_records(
                    job_id=ctx.job_id,
                    strategy_id=selected_strategy,
                    selected_rules=prov_schemas,
                    failure_analysis=prov_failure_analysis,
                    cluster_id_by_micro_rule=cluster_id_map,
                    ablation_artifact=ablation_artifact,
                )
                written_artifacts.append(write_rule_provenance_json(artifact_root, prov_artifact))
                written_artifacts.append(
                    write_rule_provenance_markdown(artifact_root, prov_artifact)
                )

        # Per-student classroom artifacts
        for _cls_strategy, _cls_student_evals in state.classroom_student_evals.items():
            written_artifacts.extend(
                write_per_student_artifacts(
                    artifact_root / "classroom", _cls_strategy, _cls_student_evals
                )
            )

        # Compiled prompt files
        compiled_prompts = state.compiled_prompts
        for strategy in DISTILLED_STRATEGIES:
            prompt_text = compiled_prompts.get(strategy)
            if prompt_text is None:
                db_synth = await get_selected_synthesized_rules_for_job(
                    self._session, ctx.job_id, strategy
                )
                schemas = [_db_synth_to_schema(r) for r in db_synth]
                prompt_text, _ = compile_prompt(ctx.task, schemas, strategy)
                compiled_prompts[strategy] = prompt_text
            written_artifacts.append(write_prompt(artifact_root, strategy, prompt_text))

        train_cases = _resolve_training_cases(
            ctx.extraction_cases,
            ctx.eval_cases,
            allow_eval_fallback=ctx.task.allow_few_shot_exemplars_from_eval_split,
        )
        for strategy in _BASELINE_FEW_SHOT_STRATEGIES:
            prompt_text = compiled_prompts.get(strategy)
            if prompt_text is None:
                selected_examples = select_deterministic_few_shot_examples(
                    train_cases,
                    k=BASELINE_FEW_SHOT_STRATEGY_TO_K[strategy],
                )
                prompt_text, _, _ = build_few_shot_prompt_with_budget(
                    baseline_prompt=state.baseline_prompt_text,
                    examples=selected_examples,
                    max_prompt_tokens=ctx.task.max_prompt_tokens,
                )
                compiled_prompts[strategy] = prompt_text
            written_artifacts.append(
                write_strategy_prompt(artifact_root, strategy, prompt_text)
            )

        if selected_strategy in compiled_prompts:
            selected_prompt_text = compiled_prompts.get(selected_strategy)
            if selected_prompt_text:
                written_artifacts.append(write_selected_prompt(artifact_root, selected_prompt_text))

        # Per-strategy debug artifacts: raw/parsed/malformed outputs, regressions vs
        # baseline, and prompt previews, for easy inspection without querying the DB.
        debug_cases_by_id = {c.id: c for c in ctx.cases}
        debug_baseline_eval = state.baseline_scaffold_eval or state.baseline_eval
        for strategy_name, strategy_eval in comparison.strategy_evals.items():
            debug_prompt_text = compiled_prompts.get(strategy_name, "")
            debug_prompt_hash = (
                sha256(debug_prompt_text.encode("utf-8")).hexdigest() if debug_prompt_text else ""
            )
            written_artifacts.extend(
                write_strategy_debug_artifacts(
                    artifact_root,
                    strategy_name,
                    strategy_eval,
                    debug_cases_by_id,
                    debug_prompt_text,
                    debug_prompt_hash,
                    baseline_eval=(
                        debug_baseline_eval if strategy_name != BASELINE_SCAFFOLD_STRATEGY else None
                    ),
                )
            )

        # ── Provenance hashes, validation, and promotion status (E/F/G) ──
        # Enriches `comparison` in place, re-writes strategy_comparison.json, and writes
        # metadata/{hashes,provenance_validation,promotion_status}.json. Best-effort: a
        # failure here never fails the job (these are reproducibility side artifacts).
        written_artifacts.extend(
            self._write_provenance_and_promotion(
                ctx, artifact_root, comparison, selected_strategy, compiled_prompts
            )
        )

        records = ctx.collector.records
        if records:
            await bulk_insert_model_call_events(self._session, ctx.job_id, records)

        usage_summary = await summarize_model_call_events(self._session, ctx.job_id)
        await update_job_usage_totals(self._session, ctx.job_id, usage_summary)
        written_artifacts.append(write_token_cost_summary(artifact_root, usage_summary))

        mlflow_run_id_path = artifact_root / "exports" / "mlflow_run_id.txt"
        if mlflow_run_id_path.exists():
            written_artifacts.append(mlflow_run_id_path)

        # Include clustering artifacts written earlier in the clustering stage (they are
        # not in `written_artifacts`, which is scoped to the export stage).
        clustering_root = artifact_root / "outputs" / "clustering"
        if clustering_root.is_dir():
            written_artifacts.extend(p for p in sorted(clustering_root.rglob("*")) if p.is_file())

        manifest_entries = _build_manifest_entries(artifact_root, written_artifacts)
        # Rich entries never include manifest.json itself: a manifest cannot carry a
        # correct hash of its own final bytes, so it is listed in `artifacts` (for legacy
        # readers) but excluded from `entries` (which verify_manifest checks by hash).
        detailed_entries = build_manifest_entries(artifact_root, written_artifacts)
        manifest_path = write_manifest(artifact_root, manifest_entries, detailed_entries)
        manifest_entries_with_manifest = sorted(
            {*manifest_entries, str(manifest_path.relative_to(artifact_root))}
        )
        write_manifest(artifact_root, manifest_entries_with_manifest, detailed_entries)

        try:
            if mlflow_run_id_path.exists():
                mlflow_run_id_for_cost = mlflow_run_id_path.read_text(encoding="utf-8").strip()
                log_artifacts_dir(
                    ctx.settings.mlflow_tracking_uri,
                    mlflow_run_id_for_cost,
                    artifact_root,
                )
                log_token_cost_metrics(
                    ctx.settings.mlflow_tracking_uri,
                    mlflow_run_id_for_cost,
                    usage_summary,
                )
        except Exception as exc:
            logger.warning("mlflow_cost_logging_failed", error=str(exc), job_id=ctx.job_id)

        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.EXPORTING_ARTIFACTS)

    def _write_provenance_and_promotion(
        self,
        ctx: PipelineContext,
        artifact_root: Path,
        comparison: StrategyComparison,
        selected_strategy: str,
        compiled_prompts: dict[str, str],
    ) -> list[Path]:
        """Compute + persist reproducibility hashes, provenance validation, promotion.

        Enriches ``comparison`` in place and re-writes strategy_comparison.json so the UI
        can read provenance rows and promotion status from a single artifact. Each check
        (hash/provenance-row construction, provenance validation, promotion evaluation,
        split-leakage validation) runs in its own try/except so one failing check never
        silently discards the others; every failure is persisted to
        metadata/integrity_check_failures.json instead of only a backend log line.
        """
        failures: list[IntegrityCheckFailure] = []

        def _record_failure(stage: str, exc: Exception) -> None:
            logger.error("integrity_check_failed", job_id=ctx.job_id, stage=stage, error=str(exc))
            failures.append(
                IntegrityCheckFailure(
                    stage=stage,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

        try:
            # Job-level hashes (safe subsets only — no secrets).
            ordered_case_ids = [case.id for case in ctx.eval_cases]
            provider_config = {
                role: route.model_dump(mode="json")
                for role, route in sorted(ctx.task.provider_model_defaults.items())
            }
            parser_config = {
                "task_mode": ctx.task.task_mode,
                "output_schema": ctx.task.output_schema,
                "allowed_evaluation_methods": sorted(ctx.task.allowed_evaluation_methods),
            }
            job_hashes = build_job_hashes(
                task=ctx.task,
                ordered_case_ids=ordered_case_ids,
                split=ctx.eval_split,
                provider_config=provider_config,
                parser_config=parser_config,
                quality_gate_config=ctx.task.quality_gates,
            )

            # Load clustering provenance captured at clustering time (not reconstructed).
            provenance_by_strategy = {
                strategy: prov
                for strategy in DISTILLED_STRATEGIES
                if (prov := load_clustering_provenance(artifact_root, strategy)) is not None
            }

            # Per-strategy provenance rows for strategy_comparison.json.
            strategy_rows = []
            for strategy in sorted(comparison.strategy_evals):
                prov = provenance_by_strategy.get(strategy)
                ref = (
                    f"outputs/clustering/{strategy}/clustering_provenance.json"
                    if prov is not None
                    else None
                )
                strategy_rows.append(
                    build_strategy_provenance_row(
                        strategy=strategy,
                        compiled_prompt=compiled_prompts.get(strategy),
                        eval_result=comparison.strategy_evals.get(strategy),
                        clustering_provenance=prov,
                        clustering_provenance_ref=ref,
                    )
                )

            artifact_hashes = build_artifact_hashes(
                job_hashes=job_hashes,
                clustering_provenance_by_strategy=provenance_by_strategy,
                strategy_rows=strategy_rows,
                selected_strategy=selected_strategy,
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a reviewable artifact below
            _record_failure("hash_and_provenance_row_construction", exc)
            report = IntegrityCheckFailuresReport(job_id=ctx.job_id, failures=failures)
            return [write_integrity_check_failures(artifact_root, report)]

        # Provenance validation. Case-set check uses sorted id sets so evaluation
        # order does not cause false positives; a strategy missing/adding cases still
        # fails. The reference is the selected strategy's set.
        prediction_case_ids = {
            strategy: sorted(case.case_id for case in ev.case_results)
            for strategy, ev in comparison.strategy_evals.items()
        }
        reference_case_ids = prediction_case_ids.get(selected_strategy, ordered_case_ids)
        prompt_hash_by_strategy = {row.strategy: row.compiled_prompt_hash for row in strategy_rows}
        validation: ProvenanceValidationReport | None
        try:
            validation = validate_provenance(
                job_id=ctx.job_id,
                artifact_root=artifact_root,
                ordered_case_ids=reference_case_ids,
                compiled_prompts=compiled_prompts,
                prompt_hash_by_strategy=prompt_hash_by_strategy,
                prediction_case_ids_by_strategy=prediction_case_ids,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed: provenance_status_from_report(None) == "missing"
            _record_failure("provenance_validation", exc)
            validation = None

        # Stale-artifact validation. Checks freshly recomputed hashes against whatever a
        # prior invocation of this job persisted to strategy_comparison.json/manifest.json.
        stale_artifact_report: StaleArtifactValidationReport | None
        try:
            stale_artifact_report = validate_stale_artifacts(
                job_id=ctx.job_id,
                artifact_root=artifact_root,
                eval_results_by_strategy=comparison.strategy_evals,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed: stale_artifact_status_from_report(None) == "missing"
            _record_failure("stale_artifact_validation", exc)
            stale_artifact_report = None

        # Promotion evaluation for the selected strategy.
        selected_gate = comparison.strategy_gates.get(selected_strategy)
        selected_eval = comparison.strategy_evals.get(selected_strategy)
        paired = comparison.paired_comparison
        selected_prov = provenance_by_strategy.get(selected_strategy)
        approved = load_approved_config(selected_strategy)
        promotion: PromotionStatus
        try:
            promotion = evaluate_promotion(
                policy=PromotionPolicy(),
                selected_strategy=selected_strategy,
                selected_gate=selected_gate,
                selected_eval=selected_eval,
                metric_delta=selected_gate.metric_delta if selected_gate else None,
                paired_fixed=(
                    selected_gate.paired_fixed
                    if selected_gate and selected_gate.paired_fixed is not None
                    else (paired.fixed_count if paired else None)
                ),
                paired_broken=(
                    selected_gate.paired_broken
                    if selected_gate and selected_gate.paired_broken is not None
                    else (paired.broken_count if paired else None)
                ),
                malformed_rate=selected_eval.malformed_output_rate if selected_eval else None,
                runtime_config_hash=(
                    clustering_config_hash(selected_prov) if selected_prov else None
                ),
                approved_config_hash=approved.config_hash if approved else None,
                provenance_report=validation,
                stale_artifact_report=stale_artifact_report,
                reproducibility_status="unverified",
                confidence_status="not_computed",
                repeat_run_count=1,
                evaluated_at=datetime.now(UTC).isoformat(),
            )
        except Exception as exc:  # noqa: BLE001 - promotion is a side artifact, not job-failing
            _record_failure("promotion_evaluation", exc)
            promotion = PromotionStatus(
                selected_strategy=selected_strategy, promotion_status="not_evaluated"
            )

        comparison.strategy_provenance = strategy_rows
        comparison.artifact_hashes = artifact_hashes
        comparison.provenance_validation = validation
        comparison.promotion_status = promotion

        leakage_report: SplitLeakageValidationReport | None
        try:
            leakage_report = validate_split_leakage(
                job_id=ctx.job_id,
                artifact_root=artifact_root,
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a reviewable artifact below
            _record_failure("split_leakage_validation", exc)
            leakage_report = None

        written = [
            write_strategy_comparison(artifact_root, comparison),
            write_artifact_hashes(artifact_root, artifact_hashes),
            write_promotion_status(artifact_root, promotion),
        ]
        if validation is not None:
            written.append(write_provenance_validation(artifact_root, validation))
        if stale_artifact_report is not None:
            written.append(write_stale_artifact_validation(artifact_root, stale_artifact_report))
        if leakage_report is not None:
            written.append(write_split_leakage_validation(artifact_root, leakage_report))
        if failures:
            report = IntegrityCheckFailuresReport(job_id=ctx.job_id, failures=failures)
            written.append(write_integrity_check_failures(artifact_root, report))
        return written
