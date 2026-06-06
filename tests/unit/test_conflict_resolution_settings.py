"""Tests for ConflictResolutionDefaults and its integration with RuleKilnTask.

Covers:
- Settings model defaults are correct
- Env vars override defaults (via nested delimiter pattern)
- Task-level None fields inherit from ConflictResolutionDefaults
- Task-level non-None fields override ConflictResolutionDefaults
- max_golden_failures = 0 is the sole "require golden pass" representation
- max_malformed_output_rate is the sole malformed-rate setting name
- No duplicate / shadow setting names exist on RuleKilnTask
"""

from __future__ import annotations

import pytest

from rulekiln.config.settings import (
    AppSettings,
    ConflictResolutionDefaults,
    QualityGateDefaults,
)
from rulekiln.schemas.task_case import RuleKilnTask

# ── ConflictResolutionDefaults defaults ──────────────────────────────────────

def test_conflict_resolution_defaults_enabled() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.enabled is True


def test_conflict_resolution_defaults_refinement_max_iterations() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.refinement_max_iterations == 3


def test_conflict_resolution_defaults_validation_holdout_fraction() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.validation_holdout_fraction == pytest.approx(0.20)


def test_conflict_resolution_defaults_max_failure_cases_per_iteration() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.max_failure_cases_per_iteration == 20


def test_conflict_resolution_defaults_max_success_cases_per_iteration() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.max_success_cases_per_iteration == 20


def test_conflict_resolution_defaults_reject_case_specific_rules() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.reject_case_specific_rules is True


def test_conflict_resolution_defaults_preserve_test_seal() -> None:
    cfg = ConflictResolutionDefaults()
    assert cfg.preserve_test_seal is True


# ── Env-var override via nested delimiter ────────────────────────────────────

def test_conflict_resolution_env_override_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_CONFLICT_RESOLUTION__REFINEMENT_MAX_ITERATIONS", "7")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlflow-test")
    settings = AppSettings()  # pyright: ignore[reportCallIssue]
    assert settings.default_conflict_resolution.refinement_max_iterations == 7


def test_conflict_resolution_env_override_holdout_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_CONFLICT_RESOLUTION__VALIDATION_HOLDOUT_FRACTION", "0.10")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlflow-test")
    settings = AppSettings()  # pyright: ignore[reportCallIssue]
    assert settings.default_conflict_resolution.validation_holdout_fraction == pytest.approx(0.10)


def test_conflict_resolution_env_override_enabled_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_CONFLICT_RESOLUTION__ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlflow-test")
    settings = AppSettings()  # pyright: ignore[reportCallIssue]
    assert settings.default_conflict_resolution.enabled is False


# ── Task-level None inherits from settings ───────────────────────────────────

def _minimal_task(**kwargs: object) -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="test",
        description="test task",
        input_template="{{ text }}",
        task_mode="classification",
        **kwargs,  # type: ignore[arg-type]
    )


def _resolve(task_val: int | None, settings_val: int) -> int:
    return task_val if task_val is not None else settings_val


def test_task_none_max_iterations_inherits_settings() -> None:
    task = _minimal_task()
    cr = ConflictResolutionDefaults()
    assert task.refinement_max_iterations is None
    assert _resolve(task.refinement_max_iterations, cr.refinement_max_iterations) == 3


def test_task_set_max_iterations_overrides_settings() -> None:
    task = _minimal_task(refinement_max_iterations=5)
    cr = ConflictResolutionDefaults()
    assert _resolve(task.refinement_max_iterations, cr.refinement_max_iterations) == 5


def test_task_none_max_failure_cases_inherits_settings() -> None:
    task = _minimal_task()
    cr = ConflictResolutionDefaults()
    assert task.refinement_max_failure_cases is None
    assert _resolve(task.refinement_max_failure_cases, cr.max_failure_cases_per_iteration) == 20


def test_task_set_max_failure_cases_overrides_settings() -> None:
    task = _minimal_task(refinement_max_failure_cases=10)
    cr = ConflictResolutionDefaults()
    assert _resolve(task.refinement_max_failure_cases, cr.max_failure_cases_per_iteration) == 10


def test_task_none_max_success_cases_inherits_settings() -> None:
    task = _minimal_task()
    cr = ConflictResolutionDefaults()
    assert task.refinement_max_success_cases is None
    assert _resolve(task.refinement_max_success_cases, cr.max_success_cases_per_iteration) == 20


def test_task_none_holdout_fraction_inherits_settings() -> None:
    task = _minimal_task()
    cr = ConflictResolutionDefaults()
    assert task.validation_holdout_fraction is None
    effective = (
        task.validation_holdout_fraction
        if task.validation_holdout_fraction is not None
        else cr.validation_holdout_fraction
    )
    assert effective == pytest.approx(0.20)


def test_task_set_holdout_fraction_overrides_settings() -> None:
    task = _minimal_task(validation_holdout_fraction=0.0)
    cr = ConflictResolutionDefaults()
    effective = (
        task.validation_holdout_fraction
        if task.validation_holdout_fraction is not None
        else cr.validation_holdout_fraction
    )
    assert effective == pytest.approx(0.0)


# ── max_golden_failures = 0 is the "require golden pass" source of truth ─────

def test_quality_gate_defaults_max_golden_failures_is_zero() -> None:
    """max_golden_failures = 0 is the sole representation of 'require golden pass'."""
    qg = QualityGateDefaults()
    assert qg.max_golden_failures == 0


def test_quality_gate_require_golden_pass_not_a_field() -> None:
    """There must be no duplicate 'require_golden_pass' field on QualityGateDefaults."""
    assert not hasattr(QualityGateDefaults(), "require_golden_pass")


def test_task_require_golden_pass_not_a_field() -> None:
    """There must be no 'require_golden_pass' field on RuleKilnTask."""
    assert not hasattr(_minimal_task(), "require_golden_pass")


# ── max_malformed_output_rate is the sole malformed-rate name ────────────────

def test_quality_gate_defaults_malformed_rate_field_name() -> None:
    """The canonical field name is max_malformed_output_rate, not max_malformed_rate."""
    qg = QualityGateDefaults()
    assert hasattr(qg, "max_malformed_output_rate")
    assert not hasattr(qg, "max_malformed_rate")
    assert qg.max_malformed_output_rate == pytest.approx(0.01)


def test_task_no_duplicate_malformed_rate_field() -> None:
    """RuleKilnTask must not expose a max_malformed_rate shadow."""
    task = _minimal_task()
    assert not hasattr(task, "max_malformed_rate")


# ── No duplicate refinement_epsilon (removed in favour of min_metric_delta) ──

def test_task_refinement_epsilon_removed() -> None:
    """refinement_epsilon was removed; min_metric_delta is the convergence threshold."""
    task = _minimal_task()
    assert not hasattr(task, "refinement_epsilon")
