"""Tests for the refinement loop's split-leakage guards and anti-memorization prompt."""

from __future__ import annotations

from rulekiln.pipeline.rule_refinement import _REFINEMENT_SYSTEM_PROMPT
from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask


def _case(case_id: str, split: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        task_mode="classification",
        input={"text": f"input {case_id}"},
        expected="positive",
        evaluation=EvaluationSpec(assertions=[]),
    )


# ── RuleKilnTask defaults ─────────────────────────────────────────────────────

def _minimal_task(**kwargs: object) -> RuleKilnTask:
    """Build a RuleKilnTask with required fields filled in."""
    return RuleKilnTask(
        task_id="t1",
        task_name="test",
        description="test task",
        input_template="{{ text }}",
        task_mode="classification",
        **kwargs,  # type: ignore[arg-type]
    )


def test_allow_refinement_on_train_split_defaults_false() -> None:
    task = _minimal_task(enable_refinement_loop=True)
    assert task.allow_refinement_on_train_split is False


def test_allow_refinement_on_train_split_can_be_set_true() -> None:
    task = _minimal_task(enable_refinement_loop=True, allow_refinement_on_train_split=True)
    assert task.allow_refinement_on_train_split is True


# ── split_policy fallback semantics ──────────────────────────────────────────

def test_split_policy_fallback_to_test_emits_warning() -> None:
    """When only test cases are present, eval falls back to test with a warning."""
    cases = [_case("test-1", "test"), _case("test-2", "test")]
    decision = resolve_split_policy(cases)

    assert decision.evaluation_split == "test"
    assert decision.fallback_warning is not None
    assert "test" in decision.fallback_warning


def test_split_policy_fallback_to_golden_emits_warning() -> None:
    """When only golden cases are present, eval falls back to golden with a warning."""
    cases = [_case("g-1", "golden")]
    decision = resolve_split_policy(cases)

    assert decision.evaluation_split == "golden"
    assert decision.fallback_warning is not None


def test_split_policy_fallback_to_train_emits_warning() -> None:
    """When there is no validation split, eval falls back to train with a warning."""
    cases = [_case("train-1", "train"), _case("train-2", "train")]
    decision = resolve_split_policy(cases)

    assert decision.evaluation_split == "train"
    assert decision.fallback_warning is not None
    assert "train" in decision.fallback_warning


def test_split_policy_no_warning_when_validation_present() -> None:
    """No fallback warning when validation cases are available."""
    cases = [_case("train-1", "train"), _case("val-1", "validation")]
    decision = resolve_split_policy(cases)

    assert decision.evaluation_split == "validation"
    assert decision.fallback_warning is None


# ── blocking-condition logic ──────────────────────────────────────────────────
# We exercise the same predicate that the worker uses so that if the worker
# logic is ever extracted to a helper it can be retested here directly.

def _refinement_blocked_reason(
    eval_split: str,
    fallback_warning: str | None,
    allow_on_train: bool,
) -> str | None:
    """Reproduce the worker's blocking predicate as a pure function for unit testing."""
    if eval_split in ("test", "golden"):
        return f"eval_split={eval_split!r}"
    if eval_split == "train" and fallback_warning and not allow_on_train:
        return "train double-use"
    return None


def test_test_split_always_blocks_refinement() -> None:
    assert _refinement_blocked_reason("test", None, allow_on_train=False) is not None
    assert _refinement_blocked_reason("test", None, allow_on_train=True) is not None


def test_golden_split_always_blocks_refinement() -> None:
    assert _refinement_blocked_reason("golden", None, allow_on_train=False) is not None
    assert _refinement_blocked_reason("golden", None, allow_on_train=True) is not None


def test_train_fallback_blocks_refinement_without_flag() -> None:
    warning = "Evaluation fell back to split=train."
    assert _refinement_blocked_reason("train", warning, allow_on_train=False) is not None


def test_train_fallback_allows_refinement_with_explicit_flag() -> None:
    warning = "Evaluation fell back to split=train."
    assert _refinement_blocked_reason("train", warning, allow_on_train=True) is None


def test_train_no_fallback_warning_does_not_block() -> None:
    """Train split without a fallback warning means train was the designated eval split."""
    assert _refinement_blocked_reason("train", None, allow_on_train=False) is None


def test_validation_split_never_blocked() -> None:
    assert _refinement_blocked_reason("validation", None, allow_on_train=False) is None
    assert _refinement_blocked_reason("validation", "some warning", allow_on_train=False) is None


# ── anti-memorization prompt ──────────────────────────────────────────────────

def test_refinement_system_prompt_contains_anti_memorization_clause() -> None:
    assert "ANTI-MEMORIZATION" in _REFINEMENT_SYSTEM_PROMPT


def test_refinement_system_prompt_forbids_case_id_references() -> None:
    prompt_lower = _REFINEMENT_SYSTEM_PROMPT.lower()
    assert "case id" in prompt_lower or "case ids" in prompt_lower


def test_refinement_system_prompt_prefers_general_rules() -> None:
    assert "general rule" in _REFINEMENT_SYSTEM_PROMPT.lower()


def test_refinement_system_prompt_flags_one_case_rules_as_memorization() -> None:
    assert "memorization" in _REFINEMENT_SYSTEM_PROMPT.lower()
