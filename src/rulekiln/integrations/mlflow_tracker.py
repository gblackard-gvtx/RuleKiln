"""MLflow integration: create runs, log params/metrics/artifacts, handle prompt registry."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rulekiln.observability.logging import get_logger

if TYPE_CHECKING:
    from rulekiln.schemas.job import DistillationRequest

logger = get_logger(__name__)


def _get_mlflow() -> ModuleType:  # pyright: ignore[reportReturnType]
    """Import mlflow lazily; raise a clear error if not installed."""
    try:
        import mlflow  # type: ignore[import-untyped]

        return mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mlflow is not installed. Add 'mlflow>=3.5,<4' to your dependencies."
        ) from exc


def create_run(
    tracking_uri: str,
    experiment_name: str,
    job_id: str,
    task_id: str,
    task_name: str,
) -> str:
    """Create an MLflow run and return its run_id."""
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{task_id}-{job_id[:8]}") as run:
        run_id: str = run.info.run_id
        mlflow.set_tags(
            {
                "job_id": job_id,
                "task_id": task_id,
                "task_name": task_name,
            }
        )

    return run_id


def log_params(tracking_uri: str, run_id: str, params: dict[str, str]) -> None:
    """Log key-value params to an existing run (batched)."""
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        # log_params accepts a flat dict[str, Any]
        mlflow.log_params(params)


def log_metrics(tracking_uri: str, run_id: str, metrics: dict[str, float]) -> None:
    """Log numeric metrics to an existing run."""
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)


def log_artifacts_dir(tracking_uri: str, run_id: str, local_dir: Path) -> None:
    """Upload an entire local directory to the MLflow run's artifact store."""
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(str(local_dir))


def log_prompt_to_registry(
    tracking_uri: str,
    run_id: str,
    prompt_name: str,
    system_prompt: str,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Optionally log a prompt to the MLflow Prompt Registry (≥3.5).

    Returns the registered model URI, or None if the registry feature is unavailable.
    """
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # MLflow ≥ 3.5 prompt registry API
        registered = mlflow.register_prompt(  # type: ignore[attr-defined]
            name=prompt_name,
            template=system_prompt,
            tags=tags or {},
        )
        uri: str = registered.uri
        logger.info("prompt_registered", prompt_name=prompt_name, uri=uri, run_id=run_id)
        return uri
    except AttributeError:
        # register_prompt not available in this MLflow build
        logger.warning("prompt_registry_unavailable", run_id=run_id)
        return None
    except Exception as exc:
        logger.warning("prompt_registry_failed", error=str(exc), run_id=run_id)
        return None


def build_run_params(
    job_id: str,
    task_id: str,
    strategy: str,
    prompt_hash: str,
) -> dict[str, str]:
    return {
        "job_id": job_id,
        "task_id": task_id,
        "selected_strategy": strategy,
        "prompt_hash": prompt_hash,
    }


def build_run_metrics(
    accuracy: float,
    macro_f1: float,
    weighted_case_score: float,
    malformed_output_rate: float,
) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_case_score": weighted_case_score,
        "malformed_output_rate": malformed_output_rate,
    }


def build_provider_params(payload: DistillationRequest) -> dict[str, str]:
    """Build per-role provider params dict suitable for mlflow.log_params."""
    params: dict[str, str] = {
        "teacher_provider_profile": payload.teacher.provider_profile,
        "teacher_model": payload.teacher.model,
        "student_provider_profile": payload.student.provider_profile,
        "student_model": payload.student.model,
        "embedding_provider_profile": payload.embedding.provider_profile,
        "embedding_model": payload.embedding.model,
    }
    if payload.judge:
        params["judge_provider_profile"] = payload.judge.provider_profile
        params["judge_model"] = payload.judge.model
    return params


def build_demo_params(
    task_id: str,
    task_mode: str,
    dataset: str,
    teacher_provider: str,
    teacher_model: str,
    student_provider: str,
    student_model: str,
    embedding_model: str,
    selected_strategy: str,
    primary_metric: str,
) -> dict[str, str]:
    """Build the minimum demo param set for a credible MLflow run."""
    return {
        "task_id": task_id,
        "task_mode": task_mode,
        "dataset": dataset,
        "teacher_provider": teacher_provider,
        "teacher_model": teacher_model,
        "student_provider": student_provider,
        "student_model": student_model,
        "embedding_model": embedding_model,
        "selected_strategy": selected_strategy,
        "primary_metric": primary_metric,
    }


def build_demo_eval_metrics(
    baseline_macro_f1: float | None,
    baseline_accuracy: float | None,
    baseline_malformed_output_rate: float | None,
    dbscan_macro_f1: float | None,
    dbscan_accuracy: float | None,
    dbscan_delta_vs_baseline: float,
    hdbscan_macro_f1: float | None,
    hdbscan_accuracy: float | None,
    hdbscan_delta_vs_baseline: float,
    selected_primary_score: float,
    selected_delta_vs_baseline: float,
    selected_passed_quality_gates: bool,
) -> dict[str, float]:
    """Build the minimum demo metric set for MLflow logging."""
    return {
        "eval.baseline.macro_f1": float(baseline_macro_f1 or 0.0),
        "eval.baseline.accuracy": float(baseline_accuracy or 0.0),
        "eval.baseline.malformed_output_rate": float(baseline_malformed_output_rate or 0.0),
        "eval.dbscan.macro_f1": float(dbscan_macro_f1 or 0.0),
        "eval.dbscan.accuracy": float(dbscan_accuracy or 0.0),
        "eval.dbscan.delta_vs_baseline": float(dbscan_delta_vs_baseline),
        "eval.hdbscan.macro_f1": float(hdbscan_macro_f1 or 0.0),
        "eval.hdbscan.accuracy": float(hdbscan_accuracy or 0.0),
        "eval.hdbscan.delta_vs_baseline": float(hdbscan_delta_vs_baseline),
        "selected.primary_score": float(selected_primary_score),
        "selected.delta_vs_baseline": float(selected_delta_vs_baseline),
        "selected.passed_quality_gates": 1.0 if selected_passed_quality_gates else 0.0,
    }


def build_token_cost_metrics(summary: dict[str, object]) -> dict[str, float]:
    """Build a flat metrics dict from a token/cost summary for MLflow logging."""
    return {
        "tokens.total": float(summary.get("total_tokens", 0)),
        "tokens.input": float(summary.get("total_input_tokens", 0)),
        "tokens.output": float(summary.get("total_output_tokens", 0)),
        "cost.total_usd": float(summary.get("estimated_total_cost_usd", 0.0)),
        "cost.teacher_usd": float(summary.get("teacher_cost_usd", 0.0)),
        "cost.student_usd": float(summary.get("student_cost_usd", 0.0)),
        "cost.embedding_usd": float(summary.get("embedding_cost_usd", 0.0)),
        "cost.judge_usd": float(summary.get("judge_cost_usd", 0.0)),
        "model_calls.total": float(summary.get("total_model_calls", 0)),
    }


def log_token_cost_metrics(tracking_uri: str, run_id: str, summary: dict[str, object]) -> None:
    """Compute and log token/cost metrics to an existing MLflow run."""
    metrics = build_token_cost_metrics(summary)
    log_metrics(tracking_uri=tracking_uri, run_id=run_id, metrics=metrics)
