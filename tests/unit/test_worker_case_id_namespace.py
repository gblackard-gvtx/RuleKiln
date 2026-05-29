"""Tests for worker-side DB case ID namespacing."""

from rulekiln.schemas.task_case import RuleKilnCase
from rulekiln.workers import distillation_worker


def test_db_case_id_is_namespaced_per_job() -> None:
    case = RuleKilnCase(
        id="banking77_validation_000000",
        split="validation",
        task_mode="classification",
        input={"utterance": "test"},
        expected={"label": "card_arrival"},
    )

    job_a = "aaaaaaaa-1111-0000-0000-000000000001"
    job_b = "bbbbbbbb-2222-0000-0000-000000000002"

    db_case_a = distillation_worker._to_db_case(job_a, case)
    db_case_b = distillation_worker._to_db_case(job_b, case)

    assert db_case_a.id != db_case_b.id
    assert db_case_a.id.startswith(f"{job_a}::")
    assert db_case_b.id.startswith(f"{job_b}::")


def test_payload_case_id_round_trip_from_namespaced_db_case_id() -> None:
    payload_case_id = "banking77_validation_000030"
    job_id = "aaaaaaaa-1111-0000-0000-000000000001"

    db_case_id = distillation_worker._db_case_id(job_id, payload_case_id)
    restored_payload_case_id = distillation_worker._payload_case_id_from_db_case_id(
        job_id,
        db_case_id,
    )

    assert restored_payload_case_id == payload_case_id
