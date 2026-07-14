"""Split-leakage detection (train/validation/test boundary enforcement).

Runs alongside :mod:`rulekiln.pipeline.provenance_validation` during artifact export
and produces a structured :class:`SplitLeakageValidationReport`. Where provenance
validation asks "are these artifacts internally hash-consistent", this module asks
a different question: "did a case from the wrong split influence this artifact".

Checks performed (see individual finding codes below):

* Train-derived artifacts (cluster assignments, rule provenance) must be built only
  from train-split cases. A declared ``source_split`` that disagrees with the actual
  split of its ``source_case_id``/``source_case_ids`` is a data-integrity error
  (``SPLIT_PROVENANCE_MISMATCH``); a train-derived artifact that legitimately traces
  back to a non-train case is contamination (``TRAIN_STAGE_NONTRAIN_CASE``).
* An evaluation artifact declared as the ``validation`` split must not reference any
  case whose actual split is not ``validation`` (``VALIDATION_EVAL_NONVALIDATION_CASE``).
* No scanned pipeline artifact may reference a test-split case
  (``TEST_CASE_LEAKAGE``) — RuleKiln has no stage today that legitimately consumes
  test-split cases before a final, explicitly-gated test evaluation.
* A refinement iteration's ``validation_tuned`` flag must agree with its
  ``refined_using_split`` value (``VALIDATION_REFINEMENT_NOT_MARKED``).
* Cases that share a source document (``metadata.document_hash``) across more than one
  split are a *warning* (``SOURCE_DOCUMENT_SPLIT_OVERLAP``), not an error: the same
  underlying document may legitimately be split at the case/hypothesis level.

Findings are read from persisted artifacts on disk (plus the in-memory case list, which
is itself the source of ``cases.normalized.jsonl``) so the check is reproducible from the
run directory alone. A missing or unparsable artifact is skipped, not an error: this
keeps the pass safe to run against legacy runs that predate ``source_split`` fields.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rulekiln.observability.logging import get_logger
from rulekiln.schemas.pipeline import EvalResult, RuleProvenanceArtifact
from rulekiln.schemas.provenance import ClusterAssignmentRecord
from rulekiln.schemas.split_leakage import SplitLeakageFinding, SplitLeakageValidationReport
from rulekiln.schemas.task_case import RuleKilnCase

_DISTILLED_STRATEGIES = ("dbscan", "hdbscan")

logger = get_logger(__name__)


def _load_cases_normalized(artifact_root: Path) -> list[RuleKilnCase]:
    path = artifact_root / "cases.normalized.jsonl"
    if not path.exists():
        return []
    cases: list[RuleKilnCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            cases.append(RuleKilnCase.model_validate_json(stripped))
    return cases


def _load_cluster_assignments(artifact_root: Path, algorithm: str) -> list[ClusterAssignmentRecord]:
    path = artifact_root / "outputs" / "clustering" / algorithm / "cluster_assignments.jsonl"
    if not path.exists():
        return []
    records: list[ClusterAssignmentRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(ClusterAssignmentRecord.model_validate_json(stripped))
    return records


def _load_rule_provenance(artifact_root: Path) -> RuleProvenanceArtifact | None:
    path = artifact_root / "outputs" / "rule_provenance.json"
    if not path.exists():
        return None
    try:
        return RuleProvenanceArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - legacy/malformed artifact, skip rather than fail
        return None


def _load_eval_results(artifact_root: Path) -> dict[str, EvalResult]:
    outputs = artifact_root / "outputs"
    if not outputs.is_dir():
        return {}
    results: dict[str, EvalResult] = {}
    for path in sorted(outputs.glob("*_eval.json")):
        try:
            results[path.stem] = EvalResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001 - legacy/malformed artifact, skip rather than fail
            logger.warning("split_leakage_eval_load_failed", path=str(path), error=str(exc))
    return results


def _load_paired_comparison_case_ids(artifact_root: Path) -> dict[str, list[str]]:
    paired_dir = artifact_root / "outputs" / "paired_comparison"
    if not paired_dir.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for name in ("fixed", "broken", "unchanged"):
        path = paired_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        ids: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                ids.append(json.loads(stripped)["case_id"])
        out[name] = ids
    return out


def _load_refinement_provenance(artifact_root: Path) -> list[dict[str, object]]:
    outputs_dir = artifact_root / "outputs"
    if not outputs_dir.is_dir():
        return []
    records: list[dict[str, object]] = []
    for path in sorted(outputs_dir.glob("rule_refinement_provenance_*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 - legacy/malformed artifact, skip rather than fail
            logger.warning(
                "split_leakage_refinement_provenance_load_failed",
                path=str(path),
                error=str(exc),
            )
    return records


def validate_split_leakage(
    *,
    job_id: str,
    artifact_root: Path,
    strategies: Sequence[str] = _DISTILLED_STRATEGIES,
) -> SplitLeakageValidationReport:
    """Cross-check persisted artifacts for train/validation/test boundary violations."""
    findings: list[SplitLeakageFinding] = []
    cases = _load_cases_normalized(artifact_root)
    split_by_case_id = {case.id: case.split for case in cases}

    def _actual_split(case_id: str) -> str | None:
        return split_by_case_id.get(case_id)

    # ── Cluster assignments: train-derived, source_split must be "train" and must
    # agree with the actual split of the case it was extracted from. ──────────────
    for algorithm in strategies:
        rel = f"outputs/clustering/{algorithm}/cluster_assignments.jsonl"
        for record in _load_cluster_assignments(artifact_root, algorithm):
            if record.source_split != "train":
                findings.append(
                    SplitLeakageFinding(
                        code="TRAIN_STAGE_NONTRAIN_CASE",
                        severity="error",
                        message=(
                            f"Cluster assignment for rule {record.rule_id!r} declares "
                            f"source_split={record.source_split!r}; clustering is a "
                            "train-only stage."
                        ),
                        expected={"source_split": "train"},
                        actual={"source_split": record.source_split},
                        artifact=rel,
                        case_ids=[record.source_case_id] if record.source_case_id else [],
                    )
                )
            if record.source_case_id is not None:
                actual = _actual_split(record.source_case_id)
                if actual is not None and actual != record.source_split:
                    findings.append(
                        SplitLeakageFinding(
                            code="SPLIT_PROVENANCE_MISMATCH",
                            severity="error",
                            message=(
                                f"Cluster assignment for rule {record.rule_id!r} declares "
                                f"source_split={record.source_split!r} but case "
                                f"{record.source_case_id!r} is actually split={actual!r}."
                            ),
                            expected={"source_split": record.source_split},
                            actual={"source_split": actual},
                            artifact=rel,
                            case_ids=[record.source_case_id],
                        )
                    )
                if actual == "test":
                    findings.append(
                        SplitLeakageFinding(
                            code="TEST_CASE_LEAKAGE",
                            severity="error",
                            message=(
                                f"Cluster assignment for rule {record.rule_id!r} references "
                                f"test-split case {record.source_case_id!r} before final test "
                                "evaluation."
                            ),
                            artifact=rel,
                            case_ids=[record.source_case_id],
                        )
                    )

    # ── Rule provenance: same train-only expectation, checked per source_case_id. ──
    rule_provenance = _load_rule_provenance(artifact_root)
    if rule_provenance is not None:
        rel = "outputs/rule_provenance.json"
        for rule in rule_provenance.rules:
            if rule.source_split != "train":
                findings.append(
                    SplitLeakageFinding(
                        code="TRAIN_STAGE_NONTRAIN_CASE",
                        severity="error",
                        message=(
                            f"Rule provenance for {rule.rule_id!r} declares "
                            f"source_split={rule.source_split!r}; rule synthesis is a "
                            "train-only stage."
                        ),
                        expected={"source_split": "train"},
                        actual={"source_split": rule.source_split},
                        artifact=rel,
                        case_ids=list(rule.source_case_ids),
                    )
                )
            for case_id in rule.source_case_ids:
                actual = _actual_split(case_id)
                if actual is not None and actual != rule.source_split:
                    findings.append(
                        SplitLeakageFinding(
                            code="SPLIT_PROVENANCE_MISMATCH",
                            severity="error",
                            message=(
                                f"Rule provenance for {rule.rule_id!r} declares "
                                f"source_split={rule.source_split!r} but case {case_id!r} is "
                                f"actually split={actual!r}."
                            ),
                            expected={"source_split": rule.source_split},
                            actual={"source_split": actual},
                            artifact=rel,
                            case_ids=[case_id],
                        )
                    )
                if actual == "test":
                    findings.append(
                        SplitLeakageFinding(
                            code="TEST_CASE_LEAKAGE",
                            severity="error",
                            message=(
                                f"Rule provenance for {rule.rule_id!r} references test-split "
                                f"case {case_id!r} before final test evaluation."
                            ),
                            artifact=rel,
                            case_ids=[case_id],
                        )
                    )

    # ── Evaluation artifacts: a declared "validation" split must contain only
    # validation cases; no scanned eval artifact may reference a test case. ────────
    for stem, eval_result in _load_eval_results(artifact_root).items():
        rel = f"outputs/{stem}.json"
        for case_result in eval_result.case_results:
            actual = _actual_split(case_result.case_id)
            if actual is None:
                continue
            if eval_result.split == "validation" and actual != "validation":
                findings.append(
                    SplitLeakageFinding(
                        code="VALIDATION_EVAL_NONVALIDATION_CASE",
                        severity="error",
                        message=(
                            f"{stem!r} is declared split='validation' but case "
                            f"{case_result.case_id!r} is actually split={actual!r}."
                        ),
                        expected={"split": "validation"},
                        actual={"split": actual},
                        artifact=rel,
                        case_ids=[case_result.case_id],
                    )
                )
            if actual == "test":
                findings.append(
                    SplitLeakageFinding(
                        code="TEST_CASE_LEAKAGE",
                        severity="error",
                        message=(
                            f"{stem!r} references test-split case {case_result.case_id!r} "
                            "before final test evaluation."
                        ),
                        artifact=rel,
                        case_ids=[case_result.case_id],
                    )
                )

    # ── Paired comparison rows: same test-leakage check (split-purity of the
    # comparison itself is already covered via the eval artifacts it derives from). ──
    for name, case_ids in _load_paired_comparison_case_ids(artifact_root).items():
        rel = f"outputs/paired_comparison/{name}.jsonl"
        for case_id in case_ids:
            if _actual_split(case_id) == "test":
                findings.append(
                    SplitLeakageFinding(
                        code="TEST_CASE_LEAKAGE",
                        severity="error",
                        message=(
                            f"Paired comparison {name!r} references test-split case "
                            f"{case_id!r} before final test evaluation."
                        ),
                        artifact=rel,
                        case_ids=[case_id],
                    )
                )

    # ── Refinement provenance: validation_tuned must agree with refined_using_split. ─
    for record in _load_refinement_provenance(artifact_root):
        refined_using_split = record.get("refined_using_split")
        validation_tuned = record.get("validation_tuned")
        if refined_using_split is None or validation_tuned is None:
            continue
        iteration = record.get("iteration")
        expected_flag = refined_using_split == "validation"
        if bool(validation_tuned) != expected_flag:
            findings.append(
                SplitLeakageFinding(
                    code="VALIDATION_REFINEMENT_NOT_MARKED",
                    severity="error",
                    message=(
                        f"Refinement iteration {iteration!r} was refined using split "
                        f"{refined_using_split!r} but validation_tuned={validation_tuned!r}."
                    ),
                    expected={"validation_tuned": expected_flag},
                    actual={"validation_tuned": validation_tuned},
                    artifact=f"outputs/rule_refinement_provenance_{iteration}.json",
                )
            )

    # ── Source-document overlap across splits: warning, not an error. ──────────────
    document_splits: dict[str, set[str]] = {}
    for case in cases:
        document_hash = case.metadata.get("document_hash")
        if isinstance(document_hash, str):
            document_splits.setdefault(document_hash, set()).add(case.split)
    for document_hash, splits in sorted(document_splits.items()):
        if len(splits) > 1:
            findings.append(
                SplitLeakageFinding(
                    code="SOURCE_DOCUMENT_SPLIT_OVERLAP",
                    severity="warning",
                    message=(
                        f"Source document {document_hash!r} has cases in multiple splits: "
                        f"{sorted(splits)}."
                    ),
                    expected={},
                    actual={"splits": sorted(splits)},
                    artifact="cases.normalized.jsonl",
                )
            )

    return SplitLeakageValidationReport(job_id=job_id, findings=findings)


def write_split_leakage_validation(root: Path, report: SplitLeakageValidationReport) -> Path:
    """Write metadata/split_leakage_validation.json."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "split_leakage_validation.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_split_leakage_validation(artifact_root: Path) -> SplitLeakageValidationReport | None:
    """Load a persisted split-leakage validation report, or ``None`` for legacy runs."""
    path = artifact_root / "metadata" / "split_leakage_validation.json"
    if not path.exists():
        return None
    return SplitLeakageValidationReport.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
