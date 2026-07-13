"""Promotion policy engine tests (Workstream M #19-26)."""

from rulekiln.pipeline.promotion import (
    BLOCKER_ORDER,
    evaluate_promotion,
    provenance_status_from_report,
)
from rulekiln.schemas.pipeline import EvalResult, QualityGateResult, RegressedLabelRow
from rulekiln.schemas.promotion import (
    MinorityLabelGuard,
    PromotionPolicy,
    ProvenanceFinding,
    ProvenanceValidationReport,
)


def _eval(strategy: str = "hdbscan", **kwargs: object) -> EvalResult:
    return EvalResult(strategy=strategy, model="m", split="validation", **kwargs)  # type: ignore[arg-type]


def _gate(passed: bool = True, **kwargs: object) -> QualityGateResult:
    return QualityGateResult(strategy="hdbscan", passed=passed, **kwargs)  # type: ignore[arg-type]


def _clean_report() -> ProvenanceValidationReport:
    return ProvenanceValidationReport(job_id="job-1", findings=[])


def _healthy_kwargs() -> dict[str, object]:
    """Inputs that would otherwise produce an approvable strategy."""
    return {
        "policy": PromotionPolicy(
            require_reproducibility=False, require_matching_approved_config=False
        ),
        "selected_strategy": "hdbscan",
        "selected_gate": _gate(),
        "selected_eval": _eval(),
        "metric_delta": 0.05,
        "paired_fixed": 29,
        "paired_broken": 18,
        "malformed_rate": 0.005,
        "runtime_config_hash": "abc",
        "approved_config_hash": "abc",
        "provenance_report": _clean_report(),
        "reproducibility_status": "reproducible",
    }


def test_selected_does_not_auto_promote() -> None:
    # #19: an approvable strategy reaches "approved", never "promoted" (promoted stays None).
    status = evaluate_promotion(**_healthy_kwargs())  # type: ignore[arg-type]
    assert status.promotion_status == "approved"
    assert status.promoted_strategy is None
    assert status.selected_strategy == "hdbscan"
    assert status.promotion_candidate == "hdbscan"


def test_invalid_provenance_blocks_promotion() -> None:
    # #20
    kwargs = _healthy_kwargs()
    kwargs["provenance_report"] = ProvenanceValidationReport(
        job_id="job-1",
        findings=[ProvenanceFinding(code="X", severity="error", message="bad")],
    )
    status = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert status.promotion_status == "blocked"
    assert status.provenance_status == "invalid"
    assert any(b.code == "PROVENANCE_INVALID" for b in status.promotion_blockers)


def test_missing_repeat_runs_blocks_promotion() -> None:
    # #21
    kwargs = _healthy_kwargs()
    kwargs["policy"] = PromotionPolicy(
        require_matching_approved_config=False, required_repeat_count=3
    )
    kwargs["reproducibility_status"] = "unverified"
    status = evaluate_promotion(**kwargs, repeat_run_count=1)  # type: ignore[arg-type]
    assert status.promotion_status == "blocked"
    assert any(b.code == "REPRODUCIBILITY_UNVERIFIED" for b in status.promotion_blockers)


def test_malformed_boundary_passes_by_default_and_blocks_when_configured() -> None:
    # #22: malformed rate exactly on threshold. Default policy does not block the boundary.
    base = _healthy_kwargs()
    base["malformed_rate"] = 0.01  # equals default max_malformed_rate
    passing = evaluate_promotion(**base)  # type: ignore[arg-type]
    assert not any("MALFORMED" in b.code for b in passing.promotion_blockers)

    strict = _healthy_kwargs()
    strict["malformed_rate"] = 0.01
    strict["policy"] = PromotionPolicy(
        require_reproducibility=False,
        require_matching_approved_config=False,
        block_on_threshold_boundary=True,
    )
    blocked = evaluate_promotion(**strict)  # type: ignore[arg-type]
    assert any(b.code == "MALFORMED_RATE_ON_BOUNDARY" for b in blocked.promotion_blockers)


def test_malformed_above_threshold_always_blocks() -> None:
    kwargs = _healthy_kwargs()
    kwargs["malformed_rate"] = 0.02
    status = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert any(b.code == "MALFORMED_RATE_ABOVE_THRESHOLD" for b in status.promotion_blockers)


def test_minority_label_regression_blocks_when_enabled() -> None:
    # #23
    regressed = RegressedLabelRow(
        label="contradiction",
        support=22,
        baseline_recall=0.77,
        candidate_recall=0.91,
        recall_delta=0.14,
        baseline_f1=0.618,
        candidate_f1=0.556,
        f1_delta=-0.062,
        new_false_negatives=0,
    )
    kwargs = _healthy_kwargs()
    kwargs["selected_eval"] = _eval(regressed_labels=[regressed])
    status = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert any(b.code == "MINORITY_LABEL_REGRESSION" for b in status.promotion_blockers)

    # Guard disabled → no blocker.
    kwargs["policy"] = PromotionPolicy(
        require_reproducibility=False,
        require_matching_approved_config=False,
        minority_label_guard=MinorityLabelGuard(enabled=False),
    )
    status_off = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert not any(b.code == "MINORITY_LABEL_REGRESSION" for b in status_off.promotion_blockers)


def test_confidence_blocker_only_when_policy_requires_it() -> None:
    # #24
    kwargs = _healthy_kwargs()
    kwargs["confidence_status"] = "insufficient"
    default_status = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert not any(
        b.code == "STATISTICAL_CONFIDENCE_INSUFFICIENT"
        for b in default_status.promotion_blockers
    )

    kwargs["policy"] = PromotionPolicy(
        require_reproducibility=False,
        require_matching_approved_config=False,
        require_confidence_check=True,
    )
    strict_status = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert any(
        b.code == "STATISTICAL_CONFIDENCE_INSUFFICIENT"
        for b in strict_status.promotion_blockers
    )


def test_config_not_promoted_config_blocks() -> None:
    kwargs = _healthy_kwargs()
    kwargs["policy"] = PromotionPolicy(require_reproducibility=False)
    kwargs["runtime_config_hash"] = "runtime-2-null"
    kwargs["approved_config_hash"] = "approved-5-1"
    status = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert any(b.code == "CONFIG_NOT_PROMOTED_CONFIG" for b in status.promotion_blockers)


def test_promotion_status_is_deterministic() -> None:
    # #25: identical inputs → identical serialized status, including blocker order.
    kwargs = _healthy_kwargs()
    kwargs["policy"] = PromotionPolicy()  # requires repro + approved config
    kwargs["reproducibility_status"] = "unverified"
    kwargs["runtime_config_hash"] = "runtime"
    kwargs["approved_config_hash"] = "approved"
    kwargs["evaluated_at"] = "2026-07-13T00:00:00+00:00"
    a = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    b = evaluate_promotion(**kwargs)  # type: ignore[arg-type]
    assert a.model_dump() == b.model_dump()
    codes = [blocker.code for blocker in a.promotion_blockers]
    assert codes == sorted(codes, key=BLOCKER_ORDER.index)


def test_provenance_status_mapping() -> None:
    assert provenance_status_from_report(None) == "missing"
    assert provenance_status_from_report(_clean_report()) == "valid"
    warn = ProvenanceValidationReport(
        findings=[ProvenanceFinding(code="W", severity="warning", message="w")]
    )
    assert provenance_status_from_report(warn) == "incomplete"
    err = ProvenanceValidationReport(
        findings=[ProvenanceFinding(code="E", severity="error", message="e")]
    )
    assert provenance_status_from_report(err) == "invalid"


def test_no_selected_strategy_is_not_evaluated() -> None:
    status = evaluate_promotion(
        policy=PromotionPolicy(),
        selected_strategy=None,
        selected_gate=None,
        selected_eval=None,
        metric_delta=None,
        paired_fixed=None,
        paired_broken=None,
        malformed_rate=None,
        runtime_config_hash=None,
        approved_config_hash=None,
        provenance_report=None,
    )
    assert status.promotion_status == "not_evaluated"
    assert status.promoted_strategy is None
