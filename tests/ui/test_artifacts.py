"""Tests: GET /ui/jobs/{job_id}/artifacts — artifact listing and download."""

import uuid
from datetime import UTC, datetime

from httpx import AsyncClient

from rulekiln.db.models import DistillationJob


async def _insert_job(factory, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    defaults = {
        "id": job_id,
        "task_id": "t1",
        "task_name": "Artifacts Task",
        "task_mode": "classification",
        "status": "completed",
        "stage": None,
        "request_json": {},
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    async with factory() as session:
        session.add(DistillationJob(**defaults))
        await session.commit()
    return defaults["id"]


class TestArtifacts:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/ui/jobs/{uuid.uuid4()}/artifacts")
        assert response.status_code == 404

    async def test_empty_artifact_dir_renders_page(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        response = await client.get(f"/ui/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        # No files available yet
        assert "No artifact files" in response.text

    async def test_artifact_file_appears_in_list(
        self, client: AsyncClient, db_session_factory, test_settings, tmp_path
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        # Create representative artifacts under tmp_path/{job_id}/...
        artifact_dir = tmp_path / job_id
        outputs_dir = artifact_dir / "outputs"
        paired_dir = outputs_dir / "paired_comparison"
        paired_dir.mkdir(parents=True)
        (artifact_dir / "task.yaml").write_text("task_id: t1\n")
        (outputs_dir / "confusion_matrix.csv").write_text(
            "actual_label,predicted_label,count\n", encoding="utf-8"
        )
        (outputs_dir / "top_confusions.md").write_text("# Top Confusions\n", encoding="utf-8")
        (paired_dir / "summary.json").write_text("{}", encoding="utf-8")

        # Override the artifact root
        test_settings.artifact_root = str(tmp_path)

        response = await client.get(f"/ui/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        assert "task.yaml" in response.text
        assert "confusion_matrix.csv" in response.text
        assert "top_confusions.md" in response.text
        assert "summary.json" in response.text

    async def test_download_path_traversal_rejected(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        response = await client.get(
            f"/ui/jobs/{job_id}/artifacts/download",
            params={"path": "../../etc/passwd"},
        )
        assert response.status_code == 400

    async def test_download_absolute_path_rejected(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        response = await client.get(
            f"/ui/jobs/{job_id}/artifacts/download",
            params={"path": "/etc/passwd"},
        )
        assert response.status_code == 400
