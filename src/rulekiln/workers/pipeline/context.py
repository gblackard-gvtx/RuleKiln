"""PipelineContext: immutable per-job context threaded through stage services."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rulekiln.config.settings import AppSettings
from rulekiln.pipeline.split_policy import SplitPolicyDecision
from rulekiln.providers.tracking import ModelCallCollector
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask


class PipelineContext(BaseModel):
    """All per-job inputs resolved before the first stage runs.

    Immutable after creation — stages read from it, never write to it.
    Mutable runtime state (eval results, compiled prompts, etc.) lives in
    PipelineArtifactsState (introduced in PR 5a).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    payload: DistillationRequest
    settings: AppSettings

    task: RuleKilnTask
    cases: list[RuleKilnCase]
    case_by_id: dict[str, RuleKilnCase]

    split_policy: SplitPolicyDecision
    extraction_cases: list[RuleKilnCase]
    eval_cases: list[RuleKilnCase]
    eval_split: str
    golden_cases: list[RuleKilnCase]

    primary_metric: str
    dataset_id: str

    collector: ModelCallCollector

    evaluation_bootstrap_enabled: bool
    evaluation_bootstrap_iterations: int
    evaluation_bootstrap_seed_offset: int
