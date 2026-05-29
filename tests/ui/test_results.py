"""Tests: GET /ui/jobs/{job_id}/results — results summary page."""

import json
import uuid
from datetime import UTC, datetime

from httpx import AsyncClient

from rulekiln.db.models import DistillationJob, EvalRun, ModelCallEvent


async def _insert_job(factory, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    defaults = {
        "id": job_id,
        "task_id": "t1",
        "task_name": "Results Task",
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


async def _insert_eval_run(
    factory,
    job_id: str,
    strategy: str,
    split: str,
    *,
    accuracy: float = 0.85,
    macro_f1: float = 0.83,
    weighted_case_score: float = 0.84,
    malformed_output_rate: float = 0.01,
) -> None:
    async with factory() as session:
        session.add(
            EvalRun(
                id=str(uuid.uuid4()),
                job_id=job_id,
                prompt_version_id=str(uuid.uuid4()),
                strategy=strategy,
                model="test-model",
                split=split,
                accuracy=accuracy,
                macro_f1=macro_f1,
                weighted_case_score=weighted_case_score,
                malformed_output_rate=malformed_output_rate,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _insert_model_call_event(
    factory,
    *,
    job_id: str,
    role: str,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
) -> None:
    async with factory() as session:
        session.add(
            ModelCallEvent(
                id=str(uuid.uuid4()),
                job_id=job_id,
                stage="reviewing_rule_conflicts",
                role=role,
                provider_profile="fake",
                provider="fake",
                model="fake-model",
                student_id=None,
                strategy="hdbscan",
                case_id=None,
                idempotency_key=f"{job_id}:{role}:{uuid.uuid4()}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                usage_estimated=False,
                input_cost_usd=0,
                output_cost_usd=cost_usd,
                total_cost_usd=cost_usd,
                cost_estimated=False,
                pricing_source="test",
                latency_ms=50,
                status="success",
                error_type=None,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


class TestResults:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/ui/jobs/{uuid.uuid4()}/results")
        assert response.status_code == 404

    async def test_no_eval_runs_renders_empty(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        response = await client.get(f"/ui/jobs/{job_id}/results")
        assert response.status_code == 200

    async def test_eval_runs_populate_metrics(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "validation")
        await _insert_eval_run(db_session_factory, job_id, "hdbscan", "validation")
        response = await client.get(f"/ui/jobs/{job_id}/results")
        assert response.status_code == 200
        # Classification defaults to macro_f1 as primary metric
        assert "macro_f1" in response.text
        assert "0.83" in response.text

    async def test_train_split_populates_metrics_when_validation_missing(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "train")
        response = await client.get(f"/ui/jobs/{job_id}/results")
        assert response.status_code == 200
        assert "macro_f1" in response.text
        assert "0.83" in response.text

    async def test_results_uses_task_primary_metric_override(
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
        response = await client.get(f"/ui/jobs/{job_id}/results")
        assert response.status_code == 200
        assert "accuracy" in response.text
        assert "0.85" in response.text

    async def test_results_uses_persisted_model_call_events_for_cost_breakdown(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(
            db_session_factory,
            selected_strategy="hdbscan",
            estimated_total_cost_usd=0.0,
            teacher_cost_usd=0.0,
            student_cost_usd=0.0,
            embedding_cost_usd=0.0,
            judge_cost_usd=0.0,
            total_tokens=0,
        )
        await _insert_eval_run(db_session_factory, job_id, "hdbscan", "train")
        await _insert_model_call_event(
            db_session_factory,
            job_id=job_id,
            role="teacher",
            cost_usd=0.3,
            input_tokens=20,
            output_tokens=10,
        )
        await _insert_model_call_event(
            db_session_factory,
            job_id=job_id,
            role="judge",
            cost_usd=0.5,
            input_tokens=40,
            output_tokens=15,
        )

        response = await client.get(f"/ui/jobs/{job_id}/results")

        assert response.status_code == 200
        assert "$0.800000" in response.text
        assert "$0.300000" in response.text
        assert "$0.500000" in response.text
        assert ">85</dd>" in response.text

    async def test_results_loads_quality_and_failure_details_from_artifacts(
        self,
        client: AsyncClient,
        db_session_factory,
        test_settings,
        tmp_path,
    ) -> None:
        test_settings.artifact_root = str(tmp_path)
        job_id = await _insert_job(db_session_factory, selected_strategy="hdbscan")

        outputs_dir = tmp_path / job_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        strategy_comparison = {
            "selected_strategy": "hdbscan",
            "hdbscan_gate": {
                "passed": True,
                "golden_failures": 2,
                "malformed_output_rate": 0.031,
            },
        }
        (outputs_dir / "strategy_comparison.json").write_text(
            json.dumps(strategy_comparison),
            encoding="utf-8",
        )
        (outputs_dir / "failures_fixed.jsonl").write_text(
            '{"case_id":"c1"}\n{"case_id":"c2"}\n',
            encoding="utf-8",
        )
        (outputs_dir / "failures_broken.jsonl").write_text(
            '{"case_id":"c3"}\n',
            encoding="utf-8",
        )

        response = await client.get(f"/ui/jobs/{job_id}/results")

        assert response.status_code == 200
        assert "Passed" in response.text
        assert "0.0310" in response.text
        assert ">2</dd>" in response.text
        assert ">1</dd>" in response.text

    async def test_results_recommendation_block_shows_lift_metrics(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, selected_strategy="hdbscan")
        await _insert_eval_run(
            db_session_factory,
            job_id,
            "baseline",
            "validation",
            accuracy=0.5314,
            macro_f1=0.1562,
            weighted_case_score=0.1562,
            malformed_output_rate=0.02,
        )
        await _insert_eval_run(
            db_session_factory,
            job_id,
            "dbscan",
            "validation",
            accuracy=0.7000,
            macro_f1=0.2800,
            weighted_case_score=0.2800,
            malformed_output_rate=0.01,
        )
        await _insert_eval_run(
            db_session_factory,
            job_id,
            "hdbscan",
            "validation",
            accuracy=0.7590,
            macro_f1=0.3245,
            weighted_case_score=0.3245,
            malformed_output_rate=0.0,
        )

        response = await client.get(f"/ui/jobs/{job_id}/results")

        assert response.status_code == 200
        assert "Best strategy" in response.text
        assert "HDBSCAN" in response.text
        assert "Baseline macro_f1" in response.text
        assert "0.1562" in response.text
        assert "Best macro_f1" in response.text
        assert "0.3245" in response.text
        assert "Delta" in response.text
        assert "+0.1683" in response.text
        assert "Relative lift" in response.text
        assert "+107.7%" in response.text
        assert "Accuracy lift" in response.text
        assert "+22.76 percentage points" in response.text
        assert "Malformed outputs" in response.text
        assert "0.00%" in response.text
