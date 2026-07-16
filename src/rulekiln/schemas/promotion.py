"""Promotion status, promotion policy, and provenance validation schemas.

These separate two distinct concepts that the integrity review found conflated:

* **Selection** (quality gates) decides whether a strategy is *eligible to be selected*.
* **Promotion** (this module's policy) decides whether an eligible selected strategy is
  *ready for reuse/deployment*.

Provenance errors block promotion but not selection (documented in
``docs/architecture/strategy_promotion.md``). All models are versioned and JSON
serializable; defaults preserve current product behavior (no significant-delta or
confidence requirement enabled by default).

Leaf module: imports only pydantic and the hashing constants.
"""

from typing import Literal

from pydantic import BaseModel, Field

from rulekiln.artifacts.canonical_hash import ALGORITHM, CANONICALIZATION_VERSION

type PromotionStatusValue = Literal[
    "not_evaluated",
    "not_applicable",
    "experimental",
    "candidate",
    "blocked",
    "approved",
    "promoted",
    "rejected",
]
type ReproducibilityStatus = Literal[
    "unknown",
    "unverified",
    "partially_verified",
    "reproducible",
    "unstable",
]
type ConfidenceStatus = Literal[
    "not_computed",
    "insufficient",
    "suggestive",
    "significant",
]
type ProvenanceStatus = Literal["missing", "incomplete", "valid", "invalid"]
type StaleArtifactStatus = Literal["missing", "valid", "invalid"]
type FindingSeverity = Literal["error", "warning"]


# ── Provenance validation (Workstream F) ──────────────────────────────────────


class ProvenanceFinding(BaseModel):
    """A structured provenance mismatch/validation finding."""

    code: str
    severity: FindingSeverity
    message: str
    expected: dict[str, object] = Field(default_factory=dict)
    actual: dict[str, object] = Field(default_factory=dict)
    artifact: str | None = None


class ProvenanceValidationReport(BaseModel):
    """Persisted provenance validation result for a job."""

    schema_version: Literal["rulekiln.provenance_validation.v1"] = (
        "rulekiln.provenance_validation.v1"
    )
    job_id: str | None = None
    findings: list[ProvenanceFinding] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(finding.severity == "warning" for finding in self.findings)


# ── Artifact hashes (Workstream D) ────────────────────────────────────────────


class ArtifactHashes(BaseModel):
    """Job/clustering/strategy-level canonical hashes persisted to metadata/hashes.json."""

    schema_version: Literal["rulekiln.artifact_hashes.v1"] = "rulekiln.artifact_hashes.v1"
    canonicalization_version: str = CANONICALIZATION_VERSION
    algorithm: str = ALGORITHM
    hashes: dict[str, str | None] = Field(default_factory=dict)


# ── Promotion policy (Workstream H) ───────────────────────────────────────────


class MinorityLabelGuard(BaseModel):
    """Guard that blocks promotion when a minority label's F1 regresses too far."""

    enabled: bool = True
    labels: list[str] = Field(default_factory=lambda: ["contradiction"])
    max_f1_regression: float = 0.02


class PromotionPolicy(BaseModel):
    """Versioned promotion policy. Defaults preserve current behavior.

    Selection is governed by quality gates elsewhere; this policy only decides whether
    an already-selected, eligible strategy is ready for promotion.
    """

    schema_version: Literal["rulekiln.promotion_policy.v1"] = "rulekiln.promotion_policy.v1"
    enabled: bool = True
    policy_version: str = "v1"

    require_selected_strategy_eligible: bool = True
    require_positive_metric_delta: bool = True
    min_metric_delta: float = 0.0

    require_paired_fixed_gt_broken: bool = True

    require_reproducibility: bool = True
    required_repeat_count: int = 3

    require_valid_provenance: bool = True
    require_no_stale_artifacts: bool = True
    require_matching_approved_config: bool = True

    # Off by default: do not require statistical significance without a product review.
    require_confidence_check: bool = False
    confidence_level: float = 0.95
    max_mcnemar_p_value: float = 0.05

    max_malformed_rate: float = 0.01
    block_on_threshold_boundary: bool = False

    minority_label_guard: MinorityLabelGuard = Field(default_factory=MinorityLabelGuard)


# ── Approved configuration identity (Workstream I) ────────────────────────────


class ApprovedClusteringConfig(BaseModel):
    """The canonical clustering configuration approved for a strategy.

    Kept independent of the clustering-provenance models so both schema modules stay
    leaves. Field order does not matter for hashing (mappings are canonicalized by key).
    """

    algorithm: str
    text_variant: str = "rule_only"
    min_cluster_size: int | None = None
    min_samples: int | None = None
    eps: float | None = None
    cluster_selection_method: str | None = None
    cluster_selection_epsilon: float | None = None
    distance_metric: str = "cosine"


class ApprovedStrategyConfig(BaseModel):
    """Registry entry identifying the approved/promoted config for a strategy."""

    schema_version: Literal["rulekiln.approved_strategy_config.v1"] = (
        "rulekiln.approved_strategy_config.v1"
    )
    strategy: str
    config: ApprovedClusteringConfig
    config_hash: str | None = None
    approved_at: str | None = None
    approved_by: str | None = None
    source_run_id: str | None = None


# ── Promotion status (Workstream G) ───────────────────────────────────────────


class PromotionBlocker(BaseModel):
    """A single reason promotion is blocked, with structured details."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class PromotionStatus(BaseModel):
    """Explicit selected/candidate/promoted semantics and blockers for a job."""

    schema_version: Literal["rulekiln.promotion_status.v1"] = "rulekiln.promotion_status.v1"
    selected_strategy: str | None = None
    selected_strategy_family: str | None = None
    promotion_candidate: str | None = None
    promoted_strategy: str | None = None
    promotion_status: PromotionStatusValue = "not_evaluated"
    promotion_policy_version: str = "v1"
    promotion_blockers: list[PromotionBlocker] = Field(default_factory=list)
    reproducibility_status: ReproducibilityStatus = "unknown"
    confidence_status: ConfidenceStatus = "not_computed"
    provenance_status: ProvenanceStatus = "missing"
    stale_artifact_status: StaleArtifactStatus = "missing"
    evaluated_at: str | None = None
    promoted_at: str | None = None
