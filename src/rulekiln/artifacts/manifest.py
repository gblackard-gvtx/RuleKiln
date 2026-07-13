"""Run manifest entries with per-artifact metadata (Workstream L).

The manifest keeps its original ``artifacts`` list of relative path strings (existing
readers depend on it) and adds a parallel ``entries`` list carrying, per artifact: type,
schema version, SHA-256, size, created timestamp, and strategy. The SHA-256 is of the
file's raw bytes so a reader can verify the manifest against the actual files on disk.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

# Path fragment → logical artifact type. First match wins; falls back to the suffix.
_TYPE_BY_FRAGMENT: tuple[tuple[str, str], ...] = (
    ("clustering_provenance.json", "clustering_provenance"),
    ("cluster_assignments.jsonl", "cluster_assignments"),
    ("cluster_summary.json", "cluster_summary"),
    ("synthesized_rules.json", "synthesized_rules"),
    ("metadata/hashes.json", "artifact_hashes"),
    ("metadata/provenance_validation.json", "provenance_validation"),
    ("metadata/promotion_status.json", "promotion_status"),
    ("strategy_comparison.json", "strategy_comparison"),
    ("eval_report.json", "eval_report"),
)

_STRATEGY_SEGMENTS = frozenset({"dbscan", "hdbscan"})


class ManifestEntry(BaseModel):
    """Metadata for one artifact in the run manifest."""

    path: str
    artifact_type: str
    schema_version: str | None = None
    sha256: str
    size_bytes: int
    created_at: str
    strategy: str | None = None


class RunManifest(BaseModel):
    """The run manifest: a legacy path list plus rich per-artifact entries."""

    schema_version: str = "rulekiln.run_manifest.v1"
    artifacts: list[str] = Field(default_factory=list)
    entries: list[ManifestEntry] = Field(default_factory=list)


def _artifact_type(rel_path: str) -> str:
    for fragment, artifact_type in _TYPE_BY_FRAGMENT:
        if rel_path.endswith(fragment) or f"/{fragment}" in rel_path:
            return artifact_type
    suffix = Path(rel_path).suffix.lstrip(".")
    return suffix or "file"


def _strategy_of(rel_path: str) -> str | None:
    for segment in rel_path.split("/"):
        if segment in _STRATEGY_SEGMENTS:
            return segment
    return None


def _schema_version_of(path: Path, rel_path: str) -> str | None:
    if path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        version = data.get("schema_version")
        return version if isinstance(version, str) else None
    return None


def build_manifest_entry(root: Path, path: Path) -> ManifestEntry | None:
    """Build a manifest entry for one file, or ``None`` if it is not under ``root``."""
    if not path.exists() or not path.is_file():
        return None
    try:
        rel_path = str(path.relative_to(root))
    except ValueError:
        return None
    raw = path.read_bytes()
    stat = path.stat()
    return ManifestEntry(
        path=rel_path,
        artifact_type=_artifact_type(rel_path),
        schema_version=_schema_version_of(path, rel_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        strategy=_strategy_of(rel_path),
    )


def build_manifest_entries(root: Path, paths: Sequence[Path]) -> list[ManifestEntry]:
    """Build sorted, de-duplicated manifest entries for the given files."""
    by_path: dict[str, ManifestEntry] = {}
    for path in paths:
        entry = build_manifest_entry(root, path)
        if entry is not None:
            by_path[entry.path] = entry
    return [by_path[key] for key in sorted(by_path)]


def verify_manifest(root: Path, manifest: RunManifest) -> list[str]:
    """Return a list of paths whose on-disk SHA-256 no longer matches the manifest."""
    mismatches: list[str] = []
    for entry in manifest.entries:
        path = root / entry.path
        if not path.exists():
            mismatches.append(entry.path)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
            mismatches.append(entry.path)
    return mismatches
