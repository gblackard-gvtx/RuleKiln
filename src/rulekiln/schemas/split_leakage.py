"""Split-leakage validation schemas.

Mirrors the ``ProvenanceFinding``/``ProvenanceValidationReport`` pattern in
``rulekiln.schemas.promotion``, but for a distinct concern: whether validation/test
cases have contaminated train-derived artifacts (or vice versa), rather than whether
persisted artifacts are internally hash-consistent.

Leaf module: imports only pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field

type SplitLeakageSeverity = Literal["error", "warning"]


class SplitLeakageFinding(BaseModel):
    """A single split-leakage finding."""

    code: str
    severity: SplitLeakageSeverity
    message: str
    expected: dict[str, object] = Field(default_factory=dict)
    actual: dict[str, object] = Field(default_factory=dict)
    artifact: str | None = None
    case_ids: list[str] = Field(default_factory=list)


class SplitLeakageValidationReport(BaseModel):
    """Persisted split-leakage validation result for a job."""

    schema_version: Literal["rulekiln.split_leakage_validation.v1"] = (
        "rulekiln.split_leakage_validation.v1"
    )
    job_id: str | None = None
    findings: list[SplitLeakageFinding] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(finding.severity == "warning" for finding in self.findings)
