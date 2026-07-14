"""BaselineEvaluationService: run the EVALUATING_BASELINE stage for one job.

Sets up prompt variants in PipelineArtifactsState, then (if the stage is not
already complete) runs the scaffold, few-shot, and embedding-based evaluations.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.eval_case_results import (
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.db.repositories.jobs import is_job_cancelled, is_stage_complete, mark_stage_complete
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
    render_few_shot_transition,
    select_deterministic_few_shot_examples,
    select_retrieval_examples,
)
from rulekiln.pipeline.evaluator import (
    JobCancelledError,
    build_case_result_from_label_prediction,
    build_eval_result_from_case_results,
    evaluate_prompt,
)
from rulekiln.pipeline.prompt_compiler import compile_baseline_prompt, count_tokens_approx
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.workers.pipeline.artifacts_state import PipelineArtifactsState
from rulekiln.workers.pipeline.classification_metrics import (
    _resolve_distance_metric_from_task,
    _resolve_training_cases,
)
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.db_mappers import (
    _build_eval_case_upsert_payload,
    _eval_case_record_to_schema,
)
from rulekiln.workers.pipeline.evaluation_persistence import (
    _insert_eval_run_if_missing,
    _persist_case_results,
)
from rulekiln.workers.pipeline.pipeline_stage import PipelineStage
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.seeding import _evaluation_bootstrap_seed
from rulekiln.workers.pipeline.stage_guard import StageGuard

BASELINE_STRATEGY = "baseline"

_log = get_logger(__name__)


async def _evaluate_strategy(
    session: AsyncSession,
    *,
    job_id: str,
    strategy: str,
    split: str,
    student_id: str,
    system_prompt: str,
    ctx: PipelineContext,
    bundle: ProviderBundle,
    user_message_prefix: str | None = None,
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

    async def _persist_case(case_result: CaseEvalResult) -> None:
        case = ctx.case_by_id.get(case_result.case_id)
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

    async def _check_cancelled() -> bool:
        return await is_job_cancelled(session, job_id)

    result = await evaluate_prompt(
        system_prompt,
        ctx.eval_cases,
        ctx.task,
        bundle.student_chat,
        bundle.student_config,
        strategy=strategy,
        split=split,
        completed_case_results=completed_case_results,
        on_case_result=_persist_case,
        user_message_prefix=user_message_prefix,
        bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
        bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
        check_cancelled=_check_cancelled,
        bootstrap_seed=_evaluation_bootstrap_seed(
            job_id=job_id,
            strategy=strategy,
            split=split,
            seed_offset=ctx.evaluation_bootstrap_seed_offset,
        ),
    )
    await _insert_eval_run_if_missing(
        session, job_id=job_id, split=split, strategy=strategy, result=result
    )
    _log.info(
        "eval_complete",
        job_id=job_id,
        strategy=strategy,
        split=split,
        malformed_output_rate=result.malformed_output_rate,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        weighted_case_score=result.weighted_case_score,
    )
    return result


class BaselineEvaluationService:
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
        from rulekiln.providers.tracking import (  # noqa: PLC0415
            ModelCallContext,
            set_tracking_context,
        )

        if ctx.payload.baseline_prompt:
            state.baseline_prompt_text = ctx.payload.baseline_prompt
            state.baseline_prompt_source = "provided"
        else:
            state.baseline_prompt_text = compile_baseline_prompt(ctx.task)
            state.baseline_prompt_source = "compiled"

        baseline_prompt_text = state.baseline_prompt_text
        baseline_prompt_source = state.baseline_prompt_source

        train_cases = _resolve_training_cases(
            ctx.extraction_cases,
            ctx.eval_cases,
            allow_eval_fallback=ctx.task.allow_few_shot_exemplars_from_eval_split,
        )
        distance_metric = _resolve_distance_metric_from_task(ctx.task)
        state.baseline_prompt_tokens = count_tokens_approx(baseline_prompt_text)
        baseline_prompt_tokens = state.baseline_prompt_tokens

        state.baseline_variant_prompts[BASELINE_SCAFFOLD_STRATEGY] = baseline_prompt_text
        state.strategy_prompt_tokens[BASELINE_SCAFFOLD_STRATEGY] = baseline_prompt_tokens
        state.strategy_metadata[BASELINE_SCAFFOLD_STRATEGY] = {
            "prompt_source": baseline_prompt_source,
        }
        state.strategy_metadata[RETRIEVAL_FEW_SHOT_STRATEGY] = {
            "configured_k": RETRIEVAL_FEW_SHOT_K,
            "distance_metric": distance_metric,
        }

        baseline_variant_prompts = state.baseline_variant_prompts
        strategy_prompt_tokens = state.strategy_prompt_tokens
        strategy_metadata = state.strategy_metadata
        baseline_variant_evals = state.baseline_variant_evals

        for strategy_name, k_value in BASELINE_FEW_SHOT_STRATEGY_TO_K.items():
            selected_examples = select_deterministic_few_shot_examples(train_cases, k=k_value)
            prompt_text, prompt_tokens, used_examples = build_few_shot_prompt_with_budget(
                baseline_prompt=baseline_prompt_text,
                examples=selected_examples,
                max_prompt_tokens=ctx.task.max_prompt_tokens,
            )
            baseline_variant_prompts[strategy_name] = prompt_text
            strategy_prompt_tokens[strategy_name] = prompt_tokens
            strategy_metadata[strategy_name] = {
                "configured_k": k_value,
                "used_examples": len(used_examples),
            }

        run_baseline_eval = phase in {"full", "evaluate_baseline"}
        if not run_baseline_eval:
            return
        if await is_stage_complete(self._session, ctx.job_id, PipelineStage.EVALUATING_BASELINE):
            return

        guard = StageGuard(self._session, ctx.job_id)
        await guard.start(PipelineStage.EVALUATING_BASELINE)

        student_profile = ctx.payload.student.provider_profile
        embedding_profile = ctx.payload.embedding.provider_profile
        eval_split = ctx.eval_split
        eval_student_id = bundle.eval_student_id

        set_tracking_context(
            ModelCallContext(
                job_id=ctx.job_id,
                stage=PipelineStage.EVALUATING_BASELINE,
                role="student",
                provider_profile=student_profile,
                provider=bundle.student_config.provider,
                model=bundle.student_config.model,
                strategy=BASELINE_STRATEGY,
            ),
            ctx.collector,
        )
        baseline_eval = await _evaluate_strategy(
            self._session,
            job_id=ctx.job_id,
            strategy=BASELINE_STRATEGY,
            split=eval_split,
            student_id=eval_student_id,
            system_prompt=baseline_prompt_text,
            ctx=ctx,
            bundle=bundle,
        )
        baseline_eval.prompt_token_count = baseline_prompt_tokens
        state.baseline_eval = baseline_eval

        baseline_scaffold_eval = baseline_eval.model_copy(
            update={
                "strategy": BASELINE_SCAFFOLD_STRATEGY,
                "prompt_token_count": baseline_prompt_tokens,
            }
        )
        state.baseline_scaffold_eval = baseline_scaffold_eval
        baseline_variant_evals[BASELINE_SCAFFOLD_STRATEGY] = baseline_scaffold_eval
        baseline_variant_prompts[BASELINE_SCAFFOLD_STRATEGY] = baseline_prompt_text
        strategy_prompt_tokens[BASELINE_SCAFFOLD_STRATEGY] = baseline_prompt_tokens
        await _insert_eval_run_if_missing(
            self._session,
            job_id=ctx.job_id,
            split=eval_split,
            strategy=BASELINE_SCAFFOLD_STRATEGY,
            result=baseline_scaffold_eval,
        )
        # Persist per-case results under baseline_scaffold so the aggregate phase can
        # load them for paired comparison. _evaluate_strategy stores under "baseline";
        # _insert_eval_run_if_missing only writes the run header, not case rows.
        await _persist_case_results(
            self._session,
            job_id=ctx.job_id,
            student_id=eval_student_id,
            strategy=BASELINE_SCAFFOLD_STRATEGY,
            split=eval_split,
            case_by_id=ctx.case_by_id,
            case_results=baseline_eval.case_results,
        )

        training_cases = _resolve_training_cases(
            ctx.extraction_cases,
            ctx.eval_cases,
            allow_eval_fallback=ctx.task.allow_few_shot_exemplars_from_eval_split,
        )

        for strategy_name, k_value in BASELINE_FEW_SHOT_STRATEGY_TO_K.items():
            selected_examples = select_deterministic_few_shot_examples(training_cases, k=k_value)
            prompt_text, prompt_tokens, used_examples = build_few_shot_prompt_with_budget(
                baseline_prompt=baseline_prompt_text,
                examples=selected_examples,
                max_prompt_tokens=ctx.task.max_prompt_tokens,
            )
            baseline_variant_prompts[strategy_name] = prompt_text
            strategy_prompt_tokens[strategy_name] = prompt_tokens
            strategy_metadata[strategy_name] = {
                "configured_k": k_value,
                "used_examples": len(used_examples),
            }
            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.EVALUATING_BASELINE,
                    role="student",
                    provider_profile=student_profile,
                    provider=bundle.student_config.provider,
                    model=bundle.student_config.model,
                    strategy=strategy_name,
                ),
                ctx.collector,
            )
            few_shot_eval = await _evaluate_strategy(
                self._session,
                job_id=ctx.job_id,
                strategy=strategy_name,
                split=eval_split,
                student_id=eval_student_id,
                system_prompt=prompt_text,
                ctx=ctx,
                bundle=bundle,
                user_message_prefix=render_few_shot_transition(len(used_examples)),
            )
            few_shot_eval.prompt_token_count = prompt_tokens
            baseline_variant_evals[strategy_name] = few_shot_eval

        labeled_train_cases = [c for c in training_cases if expected_label(c) is not None]
        _log.info(
            "embedding_baseline_setup",
            job_id=ctx.job_id,
            labeled_train_cases=len(labeled_train_cases),
            eval_cases=len(ctx.eval_cases),
        )
        if labeled_train_cases:
            train_texts = [case_text_for_embedding(c) for c in labeled_train_cases]
            train_labels = [
                expected_label(c) or ""
                for c in labeled_train_cases
            ]

            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.EVALUATING_BASELINE,
                    role="embedding",
                    provider_profile=embedding_profile,
                    provider=bundle.embedding_config.provider,
                    model=bundle.embedding_config.model,
                    strategy=EMBEDDING_CENTROID_STRATEGY,
                ),
                ctx.collector,
            )
            train_embedding_result = await bundle.embedding_client.embed_texts(
                texts=train_texts,
                config=bundle.embedding_config,
            )
            train_embeddings = train_embedding_result.embeddings

            set_tracking_context(
                ModelCallContext(
                    job_id=ctx.job_id,
                    stage=PipelineStage.EVALUATING_BASELINE,
                    role="embedding",
                    provider_profile=embedding_profile,
                    provider=bundle.embedding_config.provider,
                    model=bundle.embedding_config.model,
                    strategy=EMBEDDING_CENTROID_STRATEGY,
                ),
                ctx.collector,
            )
            eval_texts = [case_text_for_embedding(c) for c in ctx.eval_cases]
            eval_embedding_result = await bundle.embedding_client.embed_texts(
                texts=eval_texts,
                config=bundle.embedding_config,
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
                for case, prediction in zip(ctx.eval_cases, centroid_predictions, strict=True)
            ]
            await _persist_case_results(
                self._session,
                job_id=ctx.job_id,
                student_id=eval_student_id,
                strategy=EMBEDDING_CENTROID_STRATEGY,
                split=eval_split,
                case_by_id=ctx.case_by_id,
                case_results=centroid_case_results,
            )
            embedding_centroid_eval = build_eval_result_from_case_results(
                strategy=EMBEDDING_CENTROID_STRATEGY,
                model=bundle.embedding_config.model,
                split=eval_split,
                task=ctx.task,
                cases=ctx.eval_cases,
                case_results=centroid_case_results,
                prompt_token_count=baseline_prompt_tokens,
                bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                bootstrap_seed=_evaluation_bootstrap_seed(
                    job_id=ctx.job_id,
                    strategy=EMBEDDING_CENTROID_STRATEGY,
                    split=eval_split,
                    seed_offset=ctx.evaluation_bootstrap_seed_offset,
                ),
            )
            await _insert_eval_run_if_missing(
                self._session,
                job_id=ctx.job_id,
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
            _log.info(
                "eval_complete",
                job_id=ctx.job_id,
                strategy=EMBEDDING_CENTROID_STRATEGY,
                split=eval_split,
                accuracy=embedding_centroid_eval.accuracy,
                macro_f1=embedding_centroid_eval.macro_f1,
                weighted_case_score=embedding_centroid_eval.weighted_case_score,
            )

            for strategy_name, k_value in EMBEDDING_KNN_STRATEGY_TO_K.items():
                knn_predictions = predict_with_knn(
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    eval_embeddings=eval_embeddings,
                    metric=distance_metric,
                    k=k_value,
                    train_ids=[c.id for c in labeled_train_cases],
                )
                knn_case_results = [
                    build_case_result_from_label_prediction(case, prediction)
                    for case, prediction in zip(ctx.eval_cases, knn_predictions, strict=True)
                ]
                await _persist_case_results(
                    self._session,
                    job_id=ctx.job_id,
                    student_id=eval_student_id,
                    strategy=strategy_name,
                    split=eval_split,
                    case_by_id=ctx.case_by_id,
                    case_results=knn_case_results,
                )
                knn_eval = build_eval_result_from_case_results(
                    strategy=strategy_name,
                    model=bundle.embedding_config.model,
                    split=eval_split,
                    task=ctx.task,
                    cases=ctx.eval_cases,
                    case_results=knn_case_results,
                    prompt_token_count=baseline_prompt_tokens,
                    bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                    bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                    bootstrap_seed=_evaluation_bootstrap_seed(
                        job_id=ctx.job_id,
                        strategy=strategy_name,
                        split=eval_split,
                        seed_offset=ctx.evaluation_bootstrap_seed_offset,
                    ),
                )
                await _insert_eval_run_if_missing(
                    self._session,
                    job_id=ctx.job_id,
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
                _log.info(
                    "eval_complete",
                    job_id=ctx.job_id,
                    strategy=strategy_name,
                    split=eval_split,
                    accuracy=knn_eval.accuracy,
                    macro_f1=knn_eval.macro_f1,
                    weighted_case_score=knn_eval.weighted_case_score,
                )

            retrieval_case_rows = await get_eval_case_results(
                self._session,
                job_id=ctx.job_id,
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
                    existing_result.error
                    and existing_result.error.startswith("model_failure:")
                ):
                    model_failure_count += 1

            retrieval_prompt_token_peak = baseline_prompt_tokens
            for case in ctx.eval_cases:
                if case.id in retrieval_case_results:
                    continue

                if await is_job_cancelled(self._session, ctx.job_id):
                    raise JobCancelledError(
                        f"Job cancelled during retrieval-few-shot evaluation (job_id={ctx.job_id})"
                    )

                try:
                    set_tracking_context(
                        ModelCallContext(
                            job_id=ctx.job_id,
                            stage=PipelineStage.EVALUATING_BASELINE,
                            role="embedding",
                            provider_profile=embedding_profile,
                            provider=bundle.embedding_config.provider,
                            model=bundle.embedding_config.model,
                            strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                            case_id=case.id,
                        ),
                        ctx.collector,
                    )
                    query_embedding_result = await bundle.embedding_client.embed_texts(
                        texts=[case_text_for_embedding(case)],
                        config=bundle.embedding_config,
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
                        self._session,
                        job_id=ctx.job_id,
                        student_id=eval_student_id,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        split=eval_split,
                        case_by_id=ctx.case_by_id,
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
                        self._session,
                        job_id=ctx.job_id,
                        student_id=eval_student_id,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        split=eval_split,
                        case_by_id=ctx.case_by_id,
                        case_results=[retrieval_failure],
                    )
                    continue

                retrieval_prompt, retrieval_prompt_tokens, _ = build_few_shot_prompt_with_budget(
                    baseline_prompt=baseline_prompt_text,
                    examples=retrieved_examples,
                    max_prompt_tokens=ctx.task.max_prompt_tokens,
                )
                retrieval_prompt_token_peak = max(
                    retrieval_prompt_token_peak,
                    retrieval_prompt_tokens,
                )

                set_tracking_context(
                    ModelCallContext(
                        job_id=ctx.job_id,
                        stage=PipelineStage.EVALUATING_BASELINE,
                        role="student",
                        provider_profile=student_profile,
                        provider=bundle.student_config.provider,
                        model=bundle.student_config.model,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        case_id=case.id,
                    ),
                    ctx.collector,
                )
                single_case_eval = await evaluate_prompt(
                    retrieval_prompt,
                    [case],
                    ctx.task,
                    bundle.student_chat,
                    bundle.student_config,
                    strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                    split=eval_split,
                    bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                    bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                    bootstrap_seed=_evaluation_bootstrap_seed(
                        job_id=ctx.job_id,
                        strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                        split=eval_split,
                        seed_offset=ctx.evaluation_bootstrap_seed_offset,
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
                    self._session,
                    job_id=ctx.job_id,
                    student_id=eval_student_id,
                    strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                    split=eval_split,
                    case_by_id=ctx.case_by_id,
                    case_results=[case_result],
                )

            ordered_retrieval_results = [
                retrieval_case_results[case.id]
                for case in ctx.eval_cases
                if case.id in retrieval_case_results
            ]
            retrieval_eval = build_eval_result_from_case_results(
                strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                model=bundle.student_config.model,
                split=eval_split,
                task=ctx.task,
                cases=ctx.eval_cases,
                case_results=ordered_retrieval_results,
                prompt_token_count=retrieval_prompt_token_peak,
                retrieval_failure_count=retrieval_failure_count,
                model_failure_count=model_failure_count,
                bootstrap_enabled=ctx.evaluation_bootstrap_enabled,
                bootstrap_iterations=ctx.evaluation_bootstrap_iterations,
                bootstrap_seed=_evaluation_bootstrap_seed(
                    job_id=ctx.job_id,
                    strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                    split=eval_split,
                    seed_offset=ctx.evaluation_bootstrap_seed_offset,
                ),
            )
            await _insert_eval_run_if_missing(
                self._session,
                job_id=ctx.job_id,
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
            _log.info(
                "eval_complete",
                job_id=ctx.job_id,
                strategy=RETRIEVAL_FEW_SHOT_STRATEGY,
                split=eval_split,
                accuracy=retrieval_eval.accuracy,
                macro_f1=retrieval_eval.macro_f1,
                weighted_case_score=retrieval_eval.weighted_case_score,
                retrieval_failures=retrieval_failure_count,
                model_failures=model_failure_count,
            )
        else:
            from rulekiln.observability.logging import get_logger  # noqa: PLC0415

            get_logger(__name__).warning(
                "embedding_baselines_skipped",
                job_id=ctx.job_id,
                reason="No labeled training cases available for embedding baselines.",
            )

        await mark_stage_complete(self._session, ctx.job_id, PipelineStage.EVALUATING_BASELINE)
