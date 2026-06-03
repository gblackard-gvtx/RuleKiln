"""Refinement ablation: compare pipeline runs with loop ON vs loop OFF.

Produces refinement_ablation.json containing macro_f1, regression rate,
prompt token count, and teacher cost for each arm (loop_on and loop_off).
"""

from __future__ import annotations

import json
from pathlib import Path

from rulekiln.schemas.pipeline import EvalResult, RefinementAblationArtifact, RefinementAblationRow


def build_refinement_ablation(
    *,
    benchmark_name: str,
    dataset: str,
    seed: int,
    loop_off_eval: EvalResult | None,
    loop_on_eval: EvalResult | None,
    loop_off_prompt_token_count: int | None = None,
    loop_on_prompt_token_count: int | None = None,
    loop_off_teacher_cost_usd: float | None = None,
    loop_on_teacher_cost_usd: float | None = None,
    loop_off_iterations_run: int = 0,
    loop_on_iterations_run: int = 0,
    loop_off_strategy_id: str | None = None,
    loop_on_strategy_id: str | None = None,
) -> RefinementAblationArtifact:
    """Build a RefinementAblationArtifact from two EvalResult objects."""

    def _row(
        arm: str,
        eval_result: EvalResult | None,
        strategy_id: str | None,
        token_count: int | None,
        teacher_cost: float | None,
        iterations_run: int,
    ) -> RefinementAblationRow:
        macro_f1 = None
        macro_f1_ci_low = None
        macro_f1_ci_high = None
        regression_rate = None
        if eval_result is not None:
            macro_f1 = eval_result.macro_f1
            if eval_result.macro_f1_ci_95 is not None:
                macro_f1_ci_low = eval_result.macro_f1_ci_95.low
                macro_f1_ci_high = eval_result.macro_f1_ci_95.high
            if eval_result.per_label_metrics:
                total = len(eval_result.per_label_metrics)
                regressed = sum(1 for r in eval_result.regressed_labels if r.f1_delta < 0)
                regression_rate = regressed / total if total > 0 else 0.0
            token_count = token_count if token_count is not None else eval_result.prompt_token_count

        return RefinementAblationRow(
            arm=arm,  # type: ignore[arg-type]
            strategy_id=strategy_id or (eval_result.strategy if eval_result else None),
            macro_f1=macro_f1,
            macro_f1_ci_low=macro_f1_ci_low,
            macro_f1_ci_high=macro_f1_ci_high,
            regression_rate=regression_rate,
            prompt_token_count=token_count,
            teacher_cost_usd=teacher_cost,
            iterations_run=iterations_run,
        )

    loop_off_row = _row(
        "loop_off",
        loop_off_eval,
        loop_off_strategy_id,
        loop_off_prompt_token_count,
        loop_off_teacher_cost_usd,
        loop_off_iterations_run,
    )
    loop_on_row = _row(
        "loop_on",
        loop_on_eval,
        loop_on_strategy_id,
        loop_on_prompt_token_count,
        loop_on_teacher_cost_usd,
        loop_on_iterations_run,
    )

    loop_helped: bool | None = None
    delta_macro_f1: float | None = None
    if loop_off_row.macro_f1 is not None and loop_on_row.macro_f1 is not None:
        delta_macro_f1 = loop_on_row.macro_f1 - loop_off_row.macro_f1
        loop_helped = delta_macro_f1 > 0

    return RefinementAblationArtifact(
        benchmark_name=benchmark_name,
        dataset=dataset,
        seed=seed,
        rows=[loop_off_row, loop_on_row],
        loop_helped=loop_helped,
        delta_macro_f1=delta_macro_f1,
    )


def write_refinement_ablation_json(output_path: Path, artifact: RefinementAblationArtifact) -> Path:
    """Write a RefinementAblationArtifact to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2))
    return output_path


def load_eval_result_from_artifact(artifact_dir: Path, strategy: str) -> EvalResult | None:
    """Load an EvalResult from a strategy eval JSON artifact, if present."""
    candidate = artifact_dir / "outputs" / f"eval_{strategy}.json"
    if not candidate.exists():
        candidate = artifact_dir / "outputs" / "eval_report.json"
    if not candidate.exists():
        return None
    try:
        data = json.loads(candidate.read_text())
        return EvalResult.model_validate(data)
    except Exception:
        return None
