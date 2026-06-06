"""Artifact path helpers and stage marker key generators."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path


def _artifact_root(artifact_root_setting: str, job_id: str) -> Path:
    return Path(artifact_root_setting) / job_id


def _extracting_case_marker(payload_case_id: str) -> str:
    return f"extracting_case:{payload_case_id}"


def _synthesis_cluster_marker(strategy: str, rule_ids: list[str]) -> str:
    cluster_key = ",".join(sorted(rule_ids))
    digest = sha256(cluster_key.encode("utf-8")).hexdigest()[:16]
    return f"synth_cluster:{strategy}:{digest}"


def _conflict_review_rule_marker(strategy: str, synthesized_rule_id: str) -> str:
    return f"conflict_rule:{strategy}:{synthesized_rule_id}"


def _build_manifest_entries(root: Path, paths: Sequence[Path]) -> list[str]:
    entries: set[str] = set()
    for entry in paths:
        if not entry.exists():
            continue
        try:
            entries.add(str(entry.relative_to(root)))
        except ValueError:
            continue
    return sorted(entries)
