"""Refinement loop, rule ablation, and two-pass pruning optimizer helpers.

These were private helpers in distillation_worker.py, extracted here so that
late_stage_services.py can import them without creating a circular dependency.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.eval_case_results import (
    get_eval_case_results,
    upsert_eval_case_result,
)
from rulekiln.db.repositories.jobs import (
    get_selected_synthesized_rules_for_job,
    get_synthesized_rules_for_job,
    is_stage_complete,
    mark_stage_complete,
)
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.failure_analysis import (
    UNATTRIBUTED_RULE_ID,
    FailureAnalysisResult,
    analyze_failures,
)
from rulekiln.pipeline.prompt_compiler import compile_prompt
from rulekiln.pipeline.rule_pruning import PruningMode, prune_rules
from rulekiln.pipeline.rule_refinement import apply_refinements, refine_rules_with_teacher
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    PruningModeComparison,
    PruningModeRow,
    RefinementIterationArtifact,
    RuleAblationArtifact,
    RuleAblationRecord,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask
from rulekiln.workers.pipeline.db_mappers import (
    _build_eval_case_upsert_payload,
    _db_synth_to_schema,
    _eval_case_record_to_schema,
)
from rulekiln.workers.pipeline.evaluation_persistence import (
    _insert_eval_run_if_missing,
)
from rulekiln.workers.pipeline.seeding import _evaluation_bootstrap_seed
from rulekiln.workers.pipeline.strategy_utils import (
    DISTILLED_STRATEGIES,
    _primary_score_for_metric,
)

logger = get_logger(__name__)

# evaluate_prompt is imported locally to allow tests to monkeypatch it.
from rulekiln.pipeline.evaluator import evaluate_prompt  # noqa: E402


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
    """
    outputs_dir = artifact_root_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    db_synth = await get_selected_synthesized_rules_for_job(session, job_id, selected_strategy)
    initial_rules = [_db_synth_to_schema(r) for r in db_synth]
    if not initial_rules:
        return None, None

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
            logger.info("refinement_no_utility_signals", job_id=job_id, iteration=iteration)
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
                "refinement_teacher_failed", job_id=job_id, iteration=iteration, error=str(exc)
            )
            break

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
                "refinement_compile_failed", job_id=job_id, iteration=iteration, error=str(exc)
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
                "refinement_eval_failed", job_id=job_id, iteration=iteration, error=str(exc)
            )
            break

        new_metric = _primary_score_for_metric(new_eval, primary_metric)
        improvement = new_metric - current_metric

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
    """Run leave-one-rule-out ablation for the selected strategy."""
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
    """Run pass-2 optimizer (utility / utility_per_token) and produce comparison rows."""
    configured_mode: PruningMode = getattr(task, "rule_pruning_mode", "support_count")
    if configured_mode == "support_count":
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

    utility_signals: dict[str, tuple[int, int]] = {}
    if ablation_artifact is not None:
        for rec in ablation_artifact.records:
            if rec.classification == "helpful":
                fixed = abs(int(rec.changed_cases or 0))
                utility_signals[rec.rule_id] = (fixed, 0)
            elif rec.classification == "harmful":
                broken = abs(int(rec.changed_cases or 0))
                utility_signals[rec.rule_id] = (0, broken)
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
