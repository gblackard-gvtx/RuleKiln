"""Integrity check failure schema.

Persists which independent integrity check(s) (hash/provenance-row construction,
provenance validation, split-leakage validation, promotion evaluation, stale-artifact
validation) failed during artifact export, so a failure is a reviewable artifact
instead of only a backend log line. Written only when at least one check fails.

Leaf module: imports only pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field


class IntegrityCheckFailure(BaseModel):
    """A single independent integrity check that failed during artifact export."""

    stage: str
    error: str
    error_type: str
    timestamp: str


class IntegrityCheckFailuresReport(BaseModel):
    """Persisted integrity check failures for a job."""

    schema_version: Literal["rulekiln.integrity_check_failures.v1"] = (
        "rulekiln.integrity_check_failures.v1"
    )
    job_id: str | None = None
    failures: list[IntegrityCheckFailure] = Field(default_factory=list)
