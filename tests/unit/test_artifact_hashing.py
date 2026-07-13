"""Job/strategy hash aggregation tests (Workstream M #1-6 aggregation layer)."""

from rulekiln.pipeline.approved_configs import load_approved_config
from rulekiln.pipeline.artifact_hashing import (
    build_artifact_hashes,
    build_job_hashes,
    build_strategy_provenance_row,
    clustering_config_hash,
    evaluation_hash,
    ordered_case_ids_hash,
    prediction_hash,
)
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.provenance import (
    ClusteringHashes,
    ClusteringParameters,
    ClusteringProvenance,
    ClusteringSource,
    EmbeddingProvenance,
)
from rulekiln.schemas.task_case import RuleKilnTask


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="Test",
        task_mode="classification",
        description="d",
        input_template="{x}",
    )


def _job_hashes(case_ids: list[str]) -> dict[str, str | None]:
    return build_job_hashes(
        task=_task(),
        ordered_case_ids=case_ids,
        split="validation",
        provider_config={"student": {"model": "m"}},
        parser_config={"task_mode": "classification"},
        quality_gate_config={"max_malformed_output_rate": 0.01},
    )


def test_ordered_case_ids_hash_is_order_sensitive() -> None:
    # #2
    assert ordered_case_ids_hash(["a", "b", "c"]) != ordered_case_ids_hash(["c", "b", "a"])
    assert ordered_case_ids_hash(["a", "b"]) == ordered_case_ids_hash(["a", "b"])


def test_job_hashes_change_with_case_set() -> None:
    a = _job_hashes(["c1", "c2", "c3"])
    b = _job_hashes(["c1", "c2"])
    assert a["ordered_case_ids"] != b["ordered_case_ids"]
    assert a["dataset_split"] != b["dataset_split"]
    # Task config is independent of the case set.
    assert a["task_config"] == b["task_config"]


def _eval(strategy: str) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="m",
        split="validation",
        macro_f1=0.73,
        accuracy=0.79,
        case_results=[
            CaseEvalResult(case_id="c2", score=1.0, passed=True),
            CaseEvalResult(case_id="c1", score=0.0, passed=False, malformed=True),
        ],
    )


def test_prediction_and_evaluation_hash_change_with_content() -> None:
    # #5
    base = _eval("hdbscan")
    assert prediction_hash(base) == prediction_hash(_eval("hdbscan"))
    changed = _eval("hdbscan")
    changed.case_results[0].passed = False
    assert prediction_hash(base) != prediction_hash(changed)

    changed_metric = _eval("hdbscan")
    changed_metric.macro_f1 = 0.99
    assert evaluation_hash(base) != evaluation_hash(changed_metric)


def _provenance() -> ClusteringProvenance:
    return ClusteringProvenance(
        job_id="job-1",
        algorithm="hdbscan",
        text_variant="rule_only",
        effective_parameters=ClusteringParameters(
            min_cluster_size=5, min_samples=1, cluster_selection_method="eom",
            cluster_selection_epsilon=0.0,
        ),
        embedding=EmbeddingProvenance(
            provider_profile="p", provider="openai_compatible", model="mxbai", dimension=1024
        ),
        source=ClusteringSource(strategy="hdbscan"),
        hashes=ClusteringHashes(
            input_text_hash="ith",
            embedding_matrix_hash="emh",
            cluster_assignment_hash="cah",
            synthesized_rules_hash="srh",
        ),
    )


def test_clustering_config_hash_matches_approved_config() -> None:
    # The runtime effective config (5/1 eom) hashes identically to the approved registry.
    approved = load_approved_config("hdbscan")
    assert approved is not None
    assert clustering_config_hash(_provenance()) == approved.config_hash


def test_strategy_row_clustering_none_for_non_clustering_strategy() -> None:
    row = build_strategy_provenance_row(
        strategy="baseline_scaffold",
        compiled_prompt="prompt text",
        eval_result=_eval("baseline_scaffold"),
        clustering_provenance=None,
        clustering_provenance_ref=None,
    )
    assert row.clustering_config_hash is None
    assert row.cluster_assignment_hash is None
    # Prompt/prediction/evaluation hashes are still populated for non-clustering strategies.
    assert row.compiled_prompt_hash is not None
    assert row.prediction_hash is not None
    assert row.evaluation_hash is not None


def test_strategy_row_populates_clustering_hashes() -> None:
    prov = _provenance()
    row = build_strategy_provenance_row(
        strategy="hdbscan",
        compiled_prompt="prompt",
        eval_result=_eval("hdbscan"),
        clustering_provenance=prov,
        clustering_provenance_ref="outputs/clustering/hdbscan/clustering_provenance.json",
    )
    assert row.cluster_assignment_hash == "cah"
    assert row.synthesized_rules_hash == "srh"
    assert row.clustering_config_hash == clustering_config_hash(prov)


def test_artifact_hashes_selected_strategy_slots() -> None:
    job_hashes = _job_hashes(["c1", "c2"])
    prov = _provenance()
    rows = [
        build_strategy_provenance_row(
            strategy="hdbscan", compiled_prompt="p", eval_result=_eval("hdbscan"),
            clustering_provenance=prov, clustering_provenance_ref="ref",
        )
    ]
    hashes = build_artifact_hashes(
        job_hashes=job_hashes,
        clustering_provenance_by_strategy={"hdbscan": prov},
        strategy_rows=rows,
        selected_strategy="hdbscan",
    )
    assert hashes.hashes["cluster_assignments"] == "cah"
    assert hashes.hashes["embedding_matrix"] == "emh"
    assert hashes.hashes["compiled_prompt"] == rows[0].compiled_prompt_hash
    assert hashes.hashes["predictions"] == rows[0].prediction_hash
    assert hashes.hashes["task_config"] == job_hashes["task_config"]
    # Decoding config is provider-runtime and intentionally not asserted from config.
    assert hashes.hashes["decoding_config"] is None
