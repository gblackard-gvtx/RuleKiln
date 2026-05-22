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
