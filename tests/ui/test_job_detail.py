"""Tests: GET /ui/jobs/{job_id} — job detail page."""

import uuid
from datetime import UTC, datetime

from httpx import AsyncClient

from rulekiln.db.models import DistillationJob, EvalCaseResultRecord, StageMarker


async def _insert_job(factory, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    defaults = {
        "id": job_id,
        "task_id": "t1",
        "task_name": "Detail Task",
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


class TestJobDetail:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/ui/jobs/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_completed_job_renders_200(self, client: AsyncClient, db_session_factory) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "Detail Task" in response.text

    async def test_running_job_has_htmx_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "hx-get" in response.text
        assert "every 2s" in response.text

    async def test_completed_job_has_no_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "every 2s" not in response.text

    async def test_failed_job_shows_error_message(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(
            db_session_factory,
            status="failed",
            error_message="Something went wrong",
        )
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "Something went wrong" in response.text

    async def test_status_fragment_polling(self, client: AsyncClient, db_session_factory) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}/status-fragment")
        assert response.status_code == 200
        assert "hx-get" in response.text

    async def test_status_fragment_waiting_for_retry_has_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="waiting_for_retry")
        response = await client.get(f"/ui/jobs/{job_id}/status-fragment")
        assert response.status_code == 200
        assert "hx-get" in response.text

    async def test_status_fragment_completed_no_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}/status-fragment")
        assert response.status_code == 200
        assert "hx-get" not in response.text

    async def test_running_job_shows_cancel_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="cancel-pipeline-btn"' in response.text

    async def test_completed_job_hides_cancel_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="cancel-pipeline-btn"' not in response.text

    async def test_failed_retryable_job_shows_retry_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="failed_retryable")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="retry-pipeline-btn"' in response.text

    async def test_running_job_hides_retry_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="retry-pipeline-btn"' not in response.text

    async def test_job_detail_shows_teacher_and_student_progress_counts(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        request_json = {
            "task": {
                "schema_version": "rulekiln.task.v1",
                "task_id": "t1",
                "task_name": "Detail Task",
                "task_mode": "classification",
                "description": "Granularity test",
                "input_template": "{{input}}",
            },
            "cases": [
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "train-1",
                    "split": "train",
                    "task_mode": "classification",
                    "input": {"text": "a"},
                },
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "train-2",
                    "split": "train",
                    "task_mode": "classification",
                    "input": {"text": "b"},
                },
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "val-1",
                    "split": "validation",
                    "task_mode": "classification",
                    "input": {"text": "c"},
                },
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "val-2",
                    "split": "validation",
                    "task_mode": "classification",
                    "input": {"text": "d"},
                },
            ],
            "teacher": {"provider_profile": "fake_chat", "model": "teacher-model"},
            "student": {"provider_profile": "fake_chat", "model": "student-model"},
            "embedding": {"provider_profile": "fake_embed", "model": "embed-model"},
        }
        job_id = await _insert_job(
            db_session_factory,
            status="running",
            request_json=request_json,
        )

        async with db_session_factory() as session:
            session.add(
                StageMarker(
                    job_id=job_id,
                    stage="extracting_rules",
                    artifact_type="extracting_case:train-1",
                )
            )

            session.add_all(
                [
                    EvalCaseResultRecord(
                        job_id=job_id,
                        student_id="student-model",
                        strategy="baseline",
                        split="validation",
                        case_id="val-1",
                        expected_json={"label": "x"},
                        actual_json={"label": "x"},
                        assertion_scores={},
                        passed=True,
                        case_score=1.0,
                    ),
                    EvalCaseResultRecord(
                        job_id=job_id,
                        student_id="student-model",
                        strategy="baseline",
                        split="validation",
                        case_id="val-2",
                        expected_json={"label": "x"},
                        actual_json={"label": "x"},
                        assertion_scores={},
                        passed=True,
                        case_score=1.0,
                    ),
                    EvalCaseResultRecord(
                        job_id=job_id,
                        student_id="student-model",
                        strategy="dbscan",
                        split="validation",
                        case_id="val-1",
                        expected_json={"label": "x"},
                        actual_json={"label": "x"},
                        assertion_scores={},
                        passed=True,
                        case_score=1.0,
                    ),
                    EvalCaseResultRecord(
                        job_id=job_id,
                        student_id="student-model",
                        strategy="hdbscan",
                        split="validation",
                        case_id="val-1",
                        expected_json={"label": "x"},
                        actual_json={"label": "x"},
                        assertion_scores={},
                        passed=True,
                        case_score=1.0,
                    ),
                    EvalCaseResultRecord(
                        job_id=job_id,
                        student_id="student-model",
                        strategy="hdbscan",
                        split="validation",
                        case_id="val-2",
                        expected_json={"label": "x"},
                        actual_json={"label": "x"},
                        assertion_scores={},
                        passed=True,
                        case_score=1.0,
                    ),
                ]
            )
            await session.commit()

        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200

        normalized = " ".join(response.text.split())
        assert "Execution Progress" in normalized
        assert "Teacher extraction cases" in normalized
        assert "1 / 2" in normalized
        assert "Student evaluation split" in normalized
        assert "validation" in normalized
        assert "Student baseline eval cases" in normalized
        assert "Student DBSCAN eval cases" in normalized
        assert "Student HDBSCAN eval cases" in normalized
        assert "Total Cases" in normalized
        assert "Pipeline Diagnostics" in normalized

    async def test_job_detail_teacher_extraction_progress_does_not_exceed_train_total(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        request_json = {
            "task": {
                "schema_version": "rulekiln.task.v1",
                "task_id": "t1",
                "task_name": "Detail Task",
                "task_mode": "classification",
                "description": "Teacher progress cap test",
                "input_template": "{{input}}",
            },
            "cases": [
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "train-1",
                    "split": "train",
                    "task_mode": "classification",
                    "input": {"text": "a"},
                },
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "train-2",
                    "split": "train",
                    "task_mode": "classification",
                    "input": {"text": "b"},
                },
                {
                    "schema_version": "rulekiln.case.v1",
                    "id": "val-1",
                    "split": "validation",
                    "task_mode": "classification",
                    "input": {"text": "c"},
                },
            ],
            "teacher": {"provider_profile": "fake_chat", "model": "teacher-model"},
            "student": {"provider_profile": "fake_chat", "model": "student-model"},
            "embedding": {"provider_profile": "fake_embed", "model": "embed-model"},
        }
        job_id = await _insert_job(
            db_session_factory,
            status="running",
            request_json=request_json,
        )

        async with db_session_factory() as session:
            session.add_all(
                [
                    StageMarker(
                        job_id=job_id,
                        stage="extracting_rules",
                        artifact_type="extracting_case:train-1",
                    ),
                    StageMarker(
                        job_id=job_id,
                        stage="extracting_rules",
                        artifact_type="extracting_case:train-2",
                    ),
                    StageMarker(
                        job_id=job_id,
                        stage="extracting_rules",
                        artifact_type="extracting_case:val-1",
                    ),
                ]
            )
            await session.commit()

        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200

        normalized = " ".join(response.text.split())
        assert "Teacher extraction cases" in normalized
        assert "2 / 2 (100.0%)" in normalized
        assert "3 / 2" not in normalized
