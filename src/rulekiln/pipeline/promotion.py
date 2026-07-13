"""Promotion evaluation engine (Workstreams G/H/I).

Separates two concepts the integrity review found conflated:

* **Selection** (quality gates, elsewhere) — whether a strategy is *eligible to be
  selected* as this job's best result.
* **Promotion** (this module) — whether an eligible, selected strategy is *approved for
  reuse/deployment* under the configured :class:`PromotionPolicy`.

The engine is a deterministic pure function of its inputs: identical inputs always yield
an identical :class:`PromotionStatus`, and blockers are emitted in a stable order. It never
auto-promotes — reaching ``approved`` means "ready for a human to promote", and
``promoted_strategy`` stays ``None`` until an explicit approval is recorded upstream.

Legacy runs (no persisted provenance) resolve to ``experimental`` with
``provenance_status='incomplete'`` / ``reproducibility_status='unverified'`` rather than
fabricating a promotion decision.
"""

from __future__ import annotations

from collections.abc import Sequence

from rulekiln.schemas.pipeline import EvalResult, QualityGateResult
from rulekiln.schemas.promotion import (
    ConfidenceStatus,
    PromotionBlocker,
    PromotionPolicy,
    PromotionStatus,
    ProvenanceStatus,
    ProvenanceValidationReport,
    ReproducibilityStatus,
)

# Blocker codes, ordered for deterministic emission.
BLOCKER_ORDER: tuple[str, ...] = (
    "SELECTED_STRATEGY_NOT_ELIGIBLE",
    "CONFIG_NOT_PROMOTED_CONFIG",
    "METRIC_DELTA_INSUFFICIENT",
    "PAIRED_FIXED_NOT_GT_BROKEN",
    "MALFORMED_RATE_ABOVE_THRESHOLD",
    "MALFORMED_RATE_ON_BOUNDARY",
    "MINORITY_LABEL_REGRESSION",
    "STATISTICAL_CONFIDENCE_INSUFFICIENT",
    "PROVENANCE_INVALID",
    "REPRODUCIBILITY_UNVERIFIED",
)


def provenance_status_from_report(
    report: ProvenanceValidationReport | None,
) -> ProvenanceStatus:
    """Map a validation report onto a promotion provenance status.

    ``None`` (no report at all) → ``missing``; a report with error findings → ``invalid``;
    a report with only warnings → ``incomplete``; a clean report → ``valid``.
    """
    if report is None:
        return "missing"
    if report.has_errors:
        return "invalid"
    if report.has_warnings:
        return "incomplete"
    return "valid"


def minority_label_regression_blocker(
    *,
    policy: PromotionPolicy,
    selected_eval: EvalResult | None,
) -> PromotionBlocker | None:
    """Return a blocker when a guarded minority label's F1 regresses beyond tolerance."""
    guard = policy.minority_label_guard
    if not guard.enabled or selected_eval is None:
        return None
    guarded = {label.lower() for label in guard.labels}
    worst: tuple[str, float] | None = None
    for row in selected_eval.regressed_labels:
        if row.label.lower() not in guarded:
            continue
        # regressed_labels.f1_delta is candidate - baseline; a regression is negative.
        regression = -row.f1_delta
        if regression > guard.max_f1_regression and (worst is None or regression > worst[1]):
            worst = (row.label, regression)
    if worst is None:
        return None
    label, regression = worst
    return PromotionBlocker(
        code="MINORITY_LABEL_REGRESSION",
        message=(
            f"{label} F1 regressed by {regression:.4f}, beyond the configured tolerance "
            f"of {guard.max_f1_regression:.4f}."
        ),
        details={"label": label, "f1_regression": regression,
                 "max_f1_regression": guard.max_f1_regression},
    )


def evaluate_promotion(
    *,
    policy: PromotionPolicy,
    selected_strategy: str | None,
    selected_gate: QualityGateResult | None,
    selected_eval: EvalResult | None,
    metric_delta: float | None,
    paired_fixed: int | None,
    paired_broken: int | None,
    malformed_rate: float | None,
    runtime_config_hash: str | None,
    approved_config_hash: str | None,
    provenance_report: ProvenanceValidationReport | None,
    reproducibility_status: ReproducibilityStatus = "unverified",
    confidence_status: ConfidenceStatus = "not_computed",
    repeat_run_count: int = 1,
    evaluated_at: str | None = None,
) -> PromotionStatus:
    """Deterministically decide promotion status + blockers for the selected strategy.

    ``runtime_config_hash`` / ``approved_config_hash`` are ``None`` for non-clustering
    strategies (or when no approved config is registered); the approved-config check is
    then skipped rather than treated as a failure.
    """
    provenance_status = provenance_status_from_report(provenance_report)

    status = PromotionStatus(
        selected_strategy=selected_strategy,
        promotion_candidate=selected_strategy,
        promoted_strategy=None,
        promotion_policy_version=policy.policy_version,
        reproducibility_status=reproducibility_status,
        confidence_status=confidence_status,
        provenance_status=provenance_status,
        evaluated_at=evaluated_at,
    )

    if not policy.enabled:
        status.promotion_status = "not_evaluated"
        return status

    if selected_strategy is None:
        status.promotion_status = "not_evaluated"
        return status

    blockers: list[PromotionBlocker] = []

    if policy.require_selected_strategy_eligible and (
        selected_gate is None or not selected_gate.passed
    ):
        blockers.append(
            PromotionBlocker(
                code="SELECTED_STRATEGY_NOT_ELIGIBLE",
                message="Selected strategy did not pass its quality gate.",
                details={"violations": selected_gate.violations if selected_gate else []},
            )
        )

    if (
        policy.require_matching_approved_config
        and approved_config_hash is not None
        and runtime_config_hash is not None
        and runtime_config_hash != approved_config_hash
    ):
        blockers.append(
            PromotionBlocker(
                code="CONFIG_NOT_PROMOTED_CONFIG",
                message="Runtime clustering configuration does not match approved configuration.",
                details={
                    "runtime_config_hash": runtime_config_hash,
                    "approved_config_hash": approved_config_hash,
                },
            )
        )

    if policy.require_positive_metric_delta and (
        metric_delta is None or metric_delta < policy.min_metric_delta
    ):
        blockers.append(
            PromotionBlocker(
                code="METRIC_DELTA_INSUFFICIENT",
                message=(
                    f"Metric delta {metric_delta} is below the required minimum "
                    f"{policy.min_metric_delta}."
                ),
                details={"metric_delta": metric_delta, "min_metric_delta": policy.min_metric_delta},
            )
        )

    if policy.require_paired_fixed_gt_broken and (
        paired_fixed is None or paired_broken is None or paired_fixed <= paired_broken
    ):
        blockers.append(
            PromotionBlocker(
                code="PAIRED_FIXED_NOT_GT_BROKEN",
                message="Paired fixed count does not exceed broken count.",
                details={"paired_fixed": paired_fixed, "paired_broken": paired_broken},
            )
        )

    if malformed_rate is not None:
        if malformed_rate > policy.max_malformed_rate:
            blockers.append(
                PromotionBlocker(
                    code="MALFORMED_RATE_ABOVE_THRESHOLD",
                    message=(
                        f"Malformed rate {malformed_rate:.4f} exceeds the maximum allowed "
                        f"{policy.max_malformed_rate:.4f}."
                    ),
                    details={"malformed_rate": malformed_rate,
                             "max_malformed_rate": policy.max_malformed_rate},
                )
            )
        elif (
            policy.block_on_threshold_boundary
            and malformed_rate == policy.max_malformed_rate
        ):
            blockers.append(
                PromotionBlocker(
                    code="MALFORMED_RATE_ON_BOUNDARY",
                    message="Malformed rate equals the maximum allowed threshold.",
                    details={"malformed_rate": malformed_rate,
                             "max_malformed_rate": policy.max_malformed_rate},
                )
            )

    minority_blocker = minority_label_regression_blocker(
        policy=policy, selected_eval=selected_eval
    )
    if minority_blocker is not None:
        blockers.append(minority_blocker)

    if policy.require_confidence_check and confidence_status != "significant":
        blockers.append(
            PromotionBlocker(
                code="STATISTICAL_CONFIDENCE_INSUFFICIENT",
                message="Statistical confidence in the improvement is insufficient.",
                details={"confidence_status": confidence_status},
            )
        )

    if policy.require_valid_provenance and provenance_status in ("invalid", "missing"):
        blockers.append(
            PromotionBlocker(
                code="PROVENANCE_INVALID",
                message="Provenance validation reported errors or provenance is missing.",
                details={"provenance_status": provenance_status},
            )
        )

    if policy.require_reproducibility and (
        repeat_run_count < policy.required_repeat_count
        or reproducibility_status not in ("reproducible",)
    ):
        blockers.append(
            PromotionBlocker(
                code="REPRODUCIBILITY_UNVERIFIED",
                message=(
                    "Required repeat runs / reproducibility hashes were not available to "
                    "verify reproducibility."
                ),
                details={
                    "repeat_run_count": repeat_run_count,
                    "required_repeat_count": policy.required_repeat_count,
                    "reproducibility_status": reproducibility_status,
                },
            )
        )

    status.promotion_blockers = _order_blockers(blockers)
    status.promotion_status = "approved" if not blockers else "blocked"
    return status


def _order_blockers(blockers: Sequence[PromotionBlocker]) -> list[PromotionBlocker]:
    order = {code: index for index, code in enumerate(BLOCKER_ORDER)}
    return sorted(blockers, key=lambda blocker: order.get(blocker.code, len(order)))
