"""Tests: GET /ui/jobs/{job_id}/eval-report — score column metric mapping."""

import json
import uuid
from datetime import UTC, datetime

from httpx import AsyncClient

from rulekiln.db.models import DistillationJob, EvalRun


async def _insert_job(factory, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    defaults = {
        "id": job_id,
        "task_id": "t1",
        "task_name": "Eval Report Task",
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


async def _insert_eval_run(factory, job_id: str, strategy: str, split: str) -> None:
    async with factory() as session:
        session.add(
            EvalRun(
                id=str(uuid.uuid4()),
                job_id=job_id,
                prompt_version_id=str(uuid.uuid4()),
                strategy=strategy,
                model="test-model",
                split=split,
                accuracy=0.85,
                macro_f1=0.83,
                weighted_case_score=0.84,
                malformed_output_rate=0.01,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


class TestEvalReport:
    async def test_score_column_uses_macro_f1_for_classification(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "validation")

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        assert response.status_code == 200
        assert "Score (macro_f1)" in response.text
        assert "0.8300" in response.text

    async def test_score_column_honors_task_primary_metric_override(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(
            db_session_factory,
            request_json={
                "task": {
                    "evaluation": {
                        "primary_metric": "accuracy",
                    }
                }
            },
        )
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "validation")

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        assert response.status_code == 200
        assert "Score (accuracy)" in response.text
        assert "0.8500" in response.text

    async def test_eval_report_shows_validation_rows_for_all_strategies(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        await _insert_eval_run(db_session_factory, job_id, "baseline", "validation")
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "validation")
        await _insert_eval_run(db_session_factory, job_id, "hdbscan", "validation")

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        assert response.status_code == 200
        assert response.text.count(">validation<") >= 3

    async def test_eval_report_displays_split_fallback_warning(
        self,
        client: AsyncClient,
        db_session_factory,
        test_settings,
        tmp_path,
    ) -> None:
        test_settings.artifact_root = str(tmp_path)
        job_id = await _insert_job(db_session_factory)
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "train")

        outputs_dir = tmp_path / job_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        warning_message = "No validation cases detected. Evaluation fell back to split=train."
        (outputs_dir / "strategy_comparison.json").write_text(
            json.dumps(
                {
                    "evaluation_split_warning": warning_message,
                    "selected_strategy_id": "dbscan",
                    "selected_strategy_family": "distilled",
                    "best_baseline_strategy_id": "baseline_scaffold",
                    "best_distilled_strategy_id": "dbscan",
                    "paired_comparison": {
                        "fixed_count": 3,
                        "broken_count": 1,
                        "unchanged_correct_count": 2,
                        "unchanged_wrong_count": 4,
                        "net_fix_rate": 0.5,
                        "net_fix_rate_status": "ok",
                        "overall_net_fix_rate": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )

        response = await client.get(f"/ui/jobs/{job_id}/eval-report")

        assert response.status_code == 200
        assert warning_message in response.text
        assert "Strategy Comparison" in response.text
        assert "dbscan" in response.text
        assert "baseline_scaffold" in response.text
        assert "Paired fixed" in response.text
        assert "+50.00%" in response.text
