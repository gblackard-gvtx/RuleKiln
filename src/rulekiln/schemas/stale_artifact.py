"""Stale-artifact validation schemas.

Mirrors the ``SplitLeakageFinding``/``SplitLeakageValidationReport`` pattern in
``rulekiln.schemas.split_leakage``, but for a distinct concern: whether a run's freshly
recomputed hashes/predictions still match what a prior invocation of the same job
persisted to disk, rather than whether a split boundary was violated.

Leaf module: imports only pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field

type StaleArtifactSeverity = Literal["error", "warning"]


class StaleArtifactFinding(BaseModel):
    """A single stale-artifact finding."""

    code: str
    severity: StaleArtifactSeverity
    message: str
    expected: dict[str, object] = Field(default_factory=dict)
    actual: dict[str, object] = Field(default_factory=dict)
    artifact: str | None = None
    strategy: str | None = None


class StaleArtifactValidationReport(BaseModel):
    """Persisted stale-artifact validation result for a job."""

    schema_version: Literal["rulekiln.stale_artifact_validation.v1"] = (
        "rulekiln.stale_artifact_validation.v1"
    )
    job_id: str | None = None
    findings: list[StaleArtifactFinding] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(finding.severity == "warning" for finding in self.findings)
