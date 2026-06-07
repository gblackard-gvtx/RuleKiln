"""PipelineStage: canonical enum of all pipeline stage names."""

from __future__ import annotations

from enum import StrEnum


class PipelineStage(StrEnum):
    CREATED = "created"
    VALIDATING_PROJECT = "validating_project"
    EXTRACTING_RULES = "extracting_rules"
    EXTRACTING_RULES_BATCH_SUBMITTED = "extracting_rules_batch_submitted"
    EXTRACTING_RULES_BATCH_COLLECTED = "extracting_rules_batch_collected"
    EMBEDDING_RULES = "embedding_rules"
    CLUSTERING_RULES = "clustering_rules"
    SYNTHESIZING_RULES = "synthesizing_rules"
    REVIEWING_RULE_CONFLICTS = "reviewing_rule_conflicts"
    PRUNING_RULES = "pruning_rules"
    COMPILING_PROMPTS = "compiling_prompts"
    EVALUATING_BASELINE = "evaluating_baseline"
    EVALUATING_DISTILLED = "evaluating_distilled"
    SELECTING_STRATEGY = "selecting_strategy"
    ANALYZING_FAILURES = "analyzing_failures"
    REFINING_RULES = "refining_rules"
    ABLATING_RULES = "ablating_rules"
    OPTIMIZING_PRUNING = "optimizing_pruning"
    EVALUATING_CLASSROOM = "evaluating_classroom"
    CHECKING_QUALITY_GATES = "checking_quality_gates"
    LOGGING_ARTIFACTS = "logging_artifacts"
    EXPORTING_ARTIFACTS = "exporting_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"
