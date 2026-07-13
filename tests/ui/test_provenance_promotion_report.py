"""Tests: Provenance & Promotion section of GET /ui/jobs/{job_id}/eval-report (Workstream J)."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from httpx import AsyncClient

from rulekiln.db.models import DistillationJob


async def _insert_job(factory) -> str:
    job_id = str(uuid.uuid4())
    async with factory() as session:
        session.add(
            DistillationJob(
                id=job_id,
                task_id="t1",
                task_name="Prov Task",
                task_mode="classification",
                status="completed",
                stage=None,
                request_json={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        await session.commit()
    return job_id


def _write(root: Path, rel: str, payload: dict[str, object]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_full_run(root: Path, job_id: str) -> None:
    job_root = root / job_id
    _write(
        job_root,
        "outputs/strategy_comparison.json",
        {
            "selected_strategy": "hdbscan",
            "strategy_provenance": [
                {
                    "strategy": "hdbscan",
                    "family": "distilled",
                    "clustering_config_hash": "cfg-hash-abc",
                    "cluster_assignment_hash": "assign-hash-def",
                    "synthesized_rules_hash": "synth-hash-ghi",
                    "compiled_prompt_hash": "prompt-hash-jkl",
                    "prediction_hash": "pred-hash-mno",
                }
            ],
        },
    )
    _write(
        job_root,
        "outputs/clustering/hdbscan/clustering_provenance.json",
        {
            "algorithm": "hdbscan",
            "text_variant": "rule_only",
            "requested_parameters": {"min_cluster_size": 2, "min_samples": None, "eps": None,
                                     "cluster_selection_method": "eom"},
            "effective_parameters": {"min_cluster_size": 2, "min_samples": 2, "eps": None,
                                     "cluster_selection_method": "eom"},
            "embedding": {"model": "mxbai-embed-large-v1.Q6_K.gguf", "dimension": 1024},
            "output": {"cluster_count_excluding_noise": 12, "noise_pct": 34.5},
            "source": {"strategy": "hdbscan"},
        },
    )
    _write(
        job_root,
        "metadata/hashes.json",
        {"hashes": {"dataset_split": "ds-hash", "task_config": "task-hash",
                    "ordered_case_ids": "case-hash", "provider_config": "prov-hash",
                    "parser_config": "parser-hash"}},
    )
    _write(
        job_root,
        "metadata/promotion_status.json",
        {
            "selected_strategy": "hdbscan",
            "promotion_candidate": "hdbscan",
            "promoted_strategy": None,
            "promotion_status": "blocked",
            "provenance_status": "incomplete",
            "reproducibility_status": "unverified",
            "confidence_status": "insufficient",
            "promotion_blockers": [
                {"code": "CONFIG_NOT_PROMOTED_CONFIG",
                 "message": "Runtime HDBSCAN configuration does not match approved configuration.",
                 "details": {}},
                {"code": "REPRODUCIBILITY_UNVERIFIED",
                 "message": "Required repeat runs were not available.", "details": {}},
            ],
        },
    )


class TestProvenancePromotionReport:
    async def test_provenance_renders_for_new_run(
        self, client: AsyncClient, db_session_factory, test_settings, tmp_path
    ) -> None:
        # #27
        test_settings.artifact_root = str(tmp_path)
        job_id = await _insert_job(db_session_factory)
        _write_full_run(tmp_path, job_id)

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        assert response.status_code == 200
        body = response.text
        assert "Provenance &amp; Promotion" in body
        assert "cfg-hash-abc" in body  # clustering-config hash
        assert "assign-hash-def" in body  # cluster-assignment hash
        assert "prompt-hash-jkl" in body  # compiled-prompt hash
        assert "mxbai-embed-large-v1.Q6_K.gguf" in body  # embedding model
        assert "1024" in body  # dimension
        assert "rule_only" in body  # text variant
        # Requested vs effective min_samples are both shown and distinct.
        assert "min_samples=None" in body
        assert "min_samples=2" in body

    async def test_promotion_blockers_render(
        self, client: AsyncClient, db_session_factory, test_settings, tmp_path
    ) -> None:
        # #29
        test_settings.artifact_root = str(tmp_path)
        job_id = await _insert_job(db_session_factory)
        _write_full_run(tmp_path, job_id)

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        body = response.text
        assert "promotion: blocked" in body
        assert "CONFIG_NOT_PROMOTED_CONFIG" in body
        assert "does not match approved configuration" in body
        assert "REPRODUCIBILITY_UNVERIFIED" in body

    async def test_selected_candidate_promoted_not_conflated(
        self, client: AsyncClient, db_session_factory, test_settings, tmp_path
    ) -> None:
        # #30
        test_settings.artifact_root = str(tmp_path)
        job_id = await _insert_job(db_session_factory)
        _write_full_run(tmp_path, job_id)

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        body = response.text
        # The three labels are defined distinctly and promoted is explicitly "none".
        assert "Selected strategy" in body
        assert "Promotion candidate" in body
        assert "Promoted strategy" in body
        assert "does <em>not</em> mean production-ready" in body

    async def test_legacy_run_shows_unavailable(
        self, client: AsyncClient, db_session_factory, test_settings, tmp_path
    ) -> None:
        # #28: no provenance artifacts on disk.
        test_settings.artifact_root = str(tmp_path)
        job_id = await _insert_job(db_session_factory)
        (tmp_path / job_id / "outputs").mkdir(parents=True, exist_ok=True)

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        assert response.status_code == 200
        assert "Provenance unavailable for this historical run." in response.text
