"""Unit tests for PipelineStage enum — updated to include new stages."""

from rulekiln.workers.distillation_worker import PipelineStage


def test_all_required_stages_present() -> None:
    expected = {
        "created",
        "validating_project",
        "extracting_rules",
        "extracting_rules_batch_submitted",
        "extracting_rules_batch_collected",
        "embedding_rules",
        "clustering_rules",
        "synthesizing_rules",
        "reviewing_rule_conflicts",
        "pruning_rules",
        "compiling_prompts",
        "evaluating_baseline",
        "evaluating_distilled",
        "selecting_strategy",
        "analyzing_failures",
        "refining_rules",
        "ablating_rules",
        "optimizing_pruning",
        "checking_quality_gates",
        "logging_artifacts",
        "exporting_artifacts",
        "completed",
        "failed",
    }
    actual = {s.value for s in PipelineStage}
    assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"


def test_stage_count() -> None:
    assert len(PipelineStage) == 23


def test_new_stages_present() -> None:
    assert PipelineStage.REVIEWING_RULE_CONFLICTS == "reviewing_rule_conflicts"
    assert PipelineStage.PRUNING_RULES == "pruning_rules"


def test_terminal_stages() -> None:
    assert PipelineStage.COMPLETED.value == "completed"
    assert PipelineStage.FAILED.value == "failed"
