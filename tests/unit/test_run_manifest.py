"""Run manifest entry tests (Workstream M #32 manifest integrity layer)."""

import json
from pathlib import Path

from rulekiln.artifacts.manifest import (
    RunManifest,
    build_manifest_entries,
    verify_manifest,
)
from rulekiln.artifacts.writer import write_manifest


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_manifest_entries_capture_metadata(tmp_path: Path) -> None:
    prov = _write(
        tmp_path,
        "outputs/clustering/hdbscan/clustering_provenance.json",
        json.dumps({"schema_version": "rulekiln.clustering_provenance.v1"}),
    )
    task = _write(tmp_path, "task.yaml", "task_id: t1\n")

    entries = build_manifest_entries(tmp_path, [prov, task])
    by_path = {e.path: e for e in entries}

    clustering = by_path["outputs/clustering/hdbscan/clustering_provenance.json"]
    assert clustering.artifact_type == "clustering_provenance"
    assert clustering.schema_version == "rulekiln.clustering_provenance.v1"
    assert clustering.strategy == "hdbscan"
    assert clustering.size_bytes > 0
    assert len(clustering.sha256) == 64

    task_entry = by_path["task.yaml"]
    assert task_entry.artifact_type == "yaml"
    assert task_entry.schema_version is None
    assert task_entry.strategy is None


def test_write_manifest_keeps_legacy_artifacts_list_and_adds_entries(tmp_path: Path) -> None:
    path = _write(tmp_path, "metadata/hashes.json", json.dumps({"schema_version": "x"}))
    entries = build_manifest_entries(tmp_path, [path])
    manifest_path = write_manifest(tmp_path, ["metadata/hashes.json"], entries)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Legacy readers still find the flat path list.
    assert payload["artifacts"] == ["metadata/hashes.json"]
    # New readers get rich entries with a verifiable hash.
    assert payload["entries"][0]["path"] == "metadata/hashes.json"
    assert payload["entries"][0]["artifact_type"] == "artifact_hashes"


def test_verify_manifest_detects_tampering(tmp_path: Path) -> None:
    path = _write(tmp_path, "outputs/strategy_comparison.json", json.dumps({"a": 1}))
    entries = build_manifest_entries(tmp_path, [path])
    manifest = RunManifest(artifacts=["outputs/strategy_comparison.json"], entries=entries)

    assert verify_manifest(tmp_path, manifest) == []

    path.write_text(json.dumps({"a": 2}), encoding="utf-8")  # tamper
    assert verify_manifest(tmp_path, manifest) == ["outputs/strategy_comparison.json"]
