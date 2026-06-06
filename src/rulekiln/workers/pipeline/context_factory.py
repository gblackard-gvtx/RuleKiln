"""PipelineContextFactory: builds PipelineContext from a job request."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from rulekiln.config.settings import AppSettings
from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.providers.tracking import ModelCallCollector
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import RuleKilnCase, TaskMode
from rulekiln.workers.pipeline.context import PipelineContext

_KNOWN_TASK_MODES: frozenset[str] = frozenset(
    {
        "classification",
        "summarization",
        "extraction",
        "rubric_review",
        "routing",
        "tool_use",
        "freeform_generation",
        "agent_behavior",
    }
)


class PipelineContextFactory:
    """Creates a fully-populated PipelineContext from job inputs.

    All fields are computed from payload + settings — no DB access required.
    """

    @staticmethod
    def create(
        job_id: str,
        payload: DistillationRequest,
        settings: AppSettings,
    ) -> PipelineContext:
        task = payload.task
        cases = list(payload.cases)
        split_policy = resolve_split_policy(cases)

        primary_metric = _resolve_primary_metric(payload, task.task_mode)
        dataset_id = _build_dataset_id(cases)

        return PipelineContext(
            job_id=job_id,
            payload=payload,
            settings=settings,
            task=task,
            cases=cases,
            case_by_id={c.id: c for c in cases},
            split_policy=split_policy,
            extraction_cases=split_policy.extraction_cases,
            eval_cases=split_policy.evaluation_cases,
            eval_split=split_policy.evaluation_split,
            golden_cases=[c for c in cases if c.split == "golden"],
            primary_metric=primary_metric,
            dataset_id=dataset_id,
            collector=ModelCallCollector(),
            evaluation_bootstrap_enabled=settings.evaluation.bootstrap_enabled,
            evaluation_bootstrap_iterations=settings.evaluation.bootstrap_iterations,
            evaluation_bootstrap_seed_offset=settings.evaluation.bootstrap_seed_offset,
        )


def _resolve_primary_metric(payload: DistillationRequest, task_mode: str) -> str:
    if payload.metric and payload.metric.strip():
        return payload.metric.strip()
    if task_mode in _KNOWN_TASK_MODES:
        return get_primary_metric(cast(TaskMode, task_mode))
    return "weighted_case_score"


def _build_dataset_id(cases: list[RuleKilnCase]) -> str:
    sorted_cases = sorted(cases, key=lambda c: c.id)
    case_payloads = [c.model_dump(mode="json") for c in sorted_cases]
    serialized = json.dumps(case_payloads, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"cases_sha256:{digest}"
