"""Clustering parameterization: task-schema config surface and pass-through.

Guards two things introduced by the clustering-tuning work:
1. ``RuleKilnTask.clustering`` accepts a task.yaml ``clustering:`` block and
   defaults to the exact pre-existing hardcoded behavior when omitted.
2. ``cluster_dbscan`` / ``cluster_hdbscan`` accept the tuning parameters and
   keep producing the same clusters under default arguments (regression pin).
"""

from __future__ import annotations

from rulekiln.pipeline.clustering import cluster_dbscan, cluster_hdbscan
from rulekiln.schemas.task_case import ClusteringParams, RuleKilnTask


def _make_task(clustering: dict[str, dict[str, float | int | str]] | None = None) -> RuleKilnTask:
    payload: dict[str, object] = {
        "task_id": "task-clustering-params",
        "task_name": "Clustering params test",
        "task_mode": "classification",
        "description": "test",
        "input_template": "{{ input }}",
    }
    if clustering is not None:
        payload["clustering"] = clustering
    return RuleKilnTask.model_validate(payload)


def _two_blob_fixture() -> tuple[list[str], list[list[float]]]:
    """Six 4-dim vectors: two tight, mutually orthogonal blobs of three."""
    blob_a = [
        [1.0, 0.01, 0.0, 0.0],
        [1.0, 0.02, 0.0, 0.0],
        [1.0, 0.0, 0.01, 0.0],
    ]
    blob_b = [
        [0.0, 0.0, 0.01, 1.0],
        [0.0, 0.01, 0.0, 1.0],
        [0.0, 0.02, 0.02, 1.0],
    ]
    rule_ids = [f"rule-{i}" for i in range(6)]
    return rule_ids, blob_a + blob_b


def test_clustering_block_round_trips() -> None:
    task = _make_task(
        {
            "dbscan": {"eps": 0.42, "min_samples": 5},
            "hdbscan": {
                "min_cluster_size": 10,
                "min_samples": 3,
                "cluster_selection_method": "leaf",
                "cluster_selection_epsilon": 0.05,
            },
        }
    )
    assert task.clustering.dbscan.eps == 0.42
    assert task.clustering.dbscan.min_samples == 5
    assert task.clustering.hdbscan.min_cluster_size == 10
    assert task.clustering.hdbscan.min_samples == 3
    assert task.clustering.hdbscan.cluster_selection_method == "leaf"
    assert task.clustering.hdbscan.cluster_selection_epsilon == 0.05

    reparsed = RuleKilnTask.model_validate(task.model_dump())
    assert reparsed.clustering == task.clustering


def test_omitted_clustering_block_yields_current_defaults() -> None:
    task = _make_task()
    assert task.clustering == ClusteringParams()
    assert task.clustering.dbscan.eps == 0.3
    assert task.clustering.dbscan.min_samples == 2
    assert task.clustering.hdbscan.min_cluster_size == 2
    assert task.clustering.hdbscan.min_samples is None
    assert task.clustering.hdbscan.cluster_selection_method == "eom"
    assert task.clustering.hdbscan.cluster_selection_epsilon == 0.0


def test_partial_clustering_block_fills_defaults() -> None:
    task = _make_task({"dbscan": {"eps": 0.55}})
    assert task.clustering.dbscan.eps == 0.55
    assert task.clustering.dbscan.min_samples == 2
    assert task.clustering.hdbscan == ClusteringParams().hdbscan


def test_cluster_hdbscan_new_params_pass_through() -> None:
    rule_ids, embeddings = _two_blob_fixture()
    clusters = cluster_hdbscan(
        rule_ids,
        embeddings,
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.05,
    )
    assert all(c.algorithm == "hdbscan" for c in clusters)
    assert len(clusters) == 2
    grouped = {frozenset(c.rule_ids) for c in clusters}
    assert grouped == {
        frozenset({"rule-0", "rule-1", "rule-2"}),
        frozenset({"rule-3", "rule-4", "rule-5"}),
    }


def test_default_args_produce_pre_change_clusters() -> None:
    rule_ids, embeddings = _two_blob_fixture()

    dbscan_clusters = cluster_dbscan(rule_ids, embeddings)
    assert len(dbscan_clusters) == 2
    assert {frozenset(c.rule_ids) for c in dbscan_clusters} == {
        frozenset({"rule-0", "rule-1", "rule-2"}),
        frozenset({"rule-3", "rule-4", "rule-5"}),
    }

    hdbscan_clusters = cluster_hdbscan(rule_ids, embeddings)
    assert len(hdbscan_clusters) == 2
    assert {frozenset(c.rule_ids) for c in hdbscan_clusters} == {
        frozenset({"rule-0", "rule-1", "rule-2"}),
        frozenset({"rule-3", "rule-4", "rule-5"}),
    }
