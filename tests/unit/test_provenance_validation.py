"""Provenance mismatch detection tests (Workstream M #14-18)."""

from pathlib import Path

from rulekiln.artifacts.canonical_hash import hash_text
from rulekiln.artifacts.writer import (
    write_cluster_assignments,
    write_clustering_provenance,
)
from rulekiln.pipeline.artifact_hashing import clustering_config_hash
from rulekiln.pipeline.provenance_validation import validate_provenance
from rulekiln.schemas.provenance import (
    ClusterAssignmentRecord,
    ClusteringHashes,
    ClusteringParameters,
    ClusteringProvenance,
    ClusteringSource,
    EmbeddingProvenance,
)


def _embedding() -> EmbeddingProvenance:
    return EmbeddingProvenance(
        provider_profile="local_mxbai_embed",
        provider="openai_compatible",
        model="mxbai-embed-large-v1.Q6_K.gguf",
        dimension=1024,
    )


def _assignments() -> list[ClusterAssignmentRecord]:
    return [
        ClusterAssignmentRecord(
            rule_id="r1", cluster_id=0, is_noise=False, label="entailment",
            text_hash=hash_text("rule one"), embedding_row_index=0,
        ),
        ClusterAssignmentRecord(
            rule_id="r2", cluster_id=0, is_noise=False, label="neutral",
            text_hash=hash_text("rule two"), embedding_row_index=1,
        ),
    ]


def _provenance(job_id: str, assignments: list[ClusterAssignmentRecord]) -> ClusteringProvenance:
    from rulekiln.pipeline.provenance_validation import _hash_assignment_records

    return ClusteringProvenance(
        job_id=job_id,
        algorithm="hdbscan",
        text_variant="rule_only",
        requested_parameters=ClusteringParameters(min_cluster_size=2, min_samples=None),
        effective_parameters=ClusteringParameters(min_cluster_size=2, min_samples=2),
        embedding=_embedding(),
        source=ClusteringSource(strategy="hdbscan"),
        hashes=ClusteringHashes(cluster_assignment_hash=_hash_assignment_records(assignments)),
    )


def _write_run(root: Path, job_id: str, assignments: list[ClusterAssignmentRecord]) -> None:
    prov = _provenance(job_id, assignments)
    write_clustering_provenance(root, "hdbscan", prov)
    write_cluster_assignments(root, "hdbscan", assignments)


def test_runtime_task_snapshot_mismatch_produces_config_finding(tmp_path: Path) -> None:
    # #14: runtime config (2/2) differs from approved (5/1) → CLUSTER_CONFIG_MISMATCH warning.
    _write_run(tmp_path, "job-1", _assignments())
    report = validate_provenance(
        job_id="job-1",
        artifact_root=tmp_path,
        ordered_case_ids=["c1"],
        compiled_prompts={},
        prompt_hash_by_strategy={},
        prediction_case_ids_by_strategy={},
        strategies=("hdbscan",),
    )
    findings = {f.code: f for f in report.findings}
    assert "CLUSTER_CONFIG_MISMATCH" in findings
    assert findings["CLUSTER_CONFIG_MISMATCH"].severity == "warning"
    # Config mismatch alone is not an error and does not, by itself, mark provenance invalid.
    assert not report.has_errors


def test_cross_job_artifact_reference_is_error(tmp_path: Path) -> None:
    # #15
    _write_run(tmp_path, "OTHER-JOB", _assignments())
    report = validate_provenance(
        job_id="job-1",
        artifact_root=tmp_path,
        ordered_case_ids=["c1"],
        compiled_prompts={},
        prompt_hash_by_strategy={},
        prediction_case_ids_by_strategy={},
        strategies=("hdbscan",),
    )
    codes = {f.code for f in report.findings}
    assert "CROSS_JOB_ARTIFACT" in codes
    assert report.has_errors


def test_modified_prompt_fails_hash_validation(tmp_path: Path) -> None:
    # #16
    _write_run(tmp_path, "job-1", _assignments())
    report = validate_provenance(
        job_id="job-1",
        artifact_root=tmp_path,
        ordered_case_ids=["c1"],
        compiled_prompts={"hdbscan": "the ACTUAL prompt text"},
        prompt_hash_by_strategy={"hdbscan": hash_text("a DIFFERENT recorded prompt")},
        prediction_case_ids_by_strategy={},
        strategies=("hdbscan",),
    )
    codes = {f.code for f in report.findings}
    assert "PROMPT_HASH_MISMATCH" in codes
    assert report.has_errors


def test_different_case_set_fails_validation(tmp_path: Path) -> None:
    # #17
    _write_run(tmp_path, "job-1", _assignments())
    report = validate_provenance(
        job_id="job-1",
        artifact_root=tmp_path,
        ordered_case_ids=["c1", "c2", "c3"],
        compiled_prompts={},
        prompt_hash_by_strategy={},
        prediction_case_ids_by_strategy={"hdbscan": ["c1", "c2"]},  # missing c3
        strategies=("hdbscan",),
    )
    codes = {f.code for f in report.findings}
    assert "CASE_SET_MISMATCH" in codes
    assert report.has_errors


def test_assignment_hash_mismatch_is_error(tmp_path: Path) -> None:
    # #18-ish: tamper with the persisted assignments after recording the hash.
    assignments = _assignments()
    _write_run(tmp_path, "job-1", assignments)
    # Overwrite assignments file with a different record set (hash no longer matches).
    tampered = [
        ClusterAssignmentRecord(
            rule_id="r1", cluster_id=1, is_noise=False, label="contradiction",
            text_hash=hash_text("tampered"), embedding_row_index=0,
        )
    ]
    write_cluster_assignments(tmp_path, "hdbscan", tampered)
    report = validate_provenance(
        job_id="job-1",
        artifact_root=tmp_path,
        ordered_case_ids=["c1"],
        compiled_prompts={},
        prompt_hash_by_strategy={},
        prediction_case_ids_by_strategy={},
        strategies=("hdbscan",),
    )
    codes = {f.code for f in report.findings}
    assert "CLUSTER_ASSIGNMENT_HASH_MISMATCH" in codes
    assert report.has_errors


def test_clean_run_has_no_findings(tmp_path: Path) -> None:
    assignments = _assignments()
    _write_run(tmp_path, "job-1", assignments)
    # Use an approved-matching config so no config-mismatch warning appears: build a
    # provenance whose effective params equal the approved 5/1 config.
    prov = _provenance("job-1", assignments)
    prov.effective_parameters = ClusteringParameters(
        min_cluster_size=5, min_samples=1, cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
    )
    prov.requested_parameters = prov.effective_parameters
    write_clustering_provenance(tmp_path, "hdbscan", prov)
    # Sanity: this runtime config now hashes to the approved config hash.
    from rulekiln.pipeline.approved_configs import load_approved_config

    approved = load_approved_config("hdbscan")
    assert approved is not None
    assert clustering_config_hash(prov) == approved.config_hash

    report = validate_provenance(
        job_id="job-1",
        artifact_root=tmp_path,
        ordered_case_ids=["c1", "c2"],
        compiled_prompts={"hdbscan": "prompt"},
        prompt_hash_by_strategy={"hdbscan": hash_text("prompt")},
        prediction_case_ids_by_strategy={"hdbscan": ["c1", "c2"]},
        strategies=("hdbscan",),
    )
    assert report.findings == []
