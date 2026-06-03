"""Deterministic baseline strategy helpers for Phase 2 evaluations."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from rulekiln.pipeline.prompt_compiler import count_tokens_approx
from rulekiln.schemas.task_case import RuleKilnCase

BASELINE_SCAFFOLD_STRATEGY = "baseline_scaffold"
BASELINE_FEW_SHOT_STRATEGY_TO_K: dict[str, int] = {
    "baseline_few_shot_k3": 3,
    "baseline_few_shot_k5": 5,
}
EMBEDDING_CENTROID_STRATEGY = "embedding_centroid"
EMBEDDING_KNN_STRATEGY_TO_K: dict[str, int] = {
    "embedding_knn_k1": 1,
    "embedding_knn_k3": 3,
    "embedding_knn_k5": 5,
}
RETRIEVAL_FEW_SHOT_STRATEGY = "retrieval_few_shot_k5"
RETRIEVAL_FEW_SHOT_K = 5


def expected_label(case: RuleKilnCase) -> str | None:
    """Return expected label for classification/routing tasks when available."""
    if case.task_mode not in {"classification", "routing"}:
        return None
    if isinstance(case.expected, dict):
        raw_label = case.expected.get("label")
        if isinstance(raw_label, str) and raw_label.strip():
            return raw_label.strip()
        return None
    if isinstance(case.expected, str) and case.expected.strip():
        return case.expected.strip()
    return None


def case_text_for_embedding(case: RuleKilnCase) -> str:
    """Build stable embedding text from case input."""
    return json.dumps(case.input, ensure_ascii=False, sort_keys=True)


def select_deterministic_few_shot_examples(
    train_cases: list[RuleKilnCase],
    *,
    k: int,
) -> list[RuleKilnCase]:
    """Select up to k examples in deterministic, label-balanced round-robin order."""
    if k <= 0:
        return []

    grouped: dict[str, list[RuleKilnCase]] = defaultdict(list)
    fallback_cases: list[RuleKilnCase] = []
    for case in sorted(train_cases, key=lambda item: item.id):
        label = expected_label(case)
        if label is None:
            fallback_cases.append(case)
            continue
        grouped[label].append(case)

    if not grouped:
        return fallback_cases[:k]

    labels = sorted(grouped.keys())
    indices = dict.fromkeys(labels, 0)
    selected: list[RuleKilnCase] = []

    while len(selected) < k:
        progressed = False
        for label in labels:
            idx = indices[label]
            cases_for_label = grouped[label]
            if idx >= len(cases_for_label):
                continue
            selected.append(cases_for_label[idx])
            indices[label] = idx + 1
            progressed = True
            if len(selected) >= k:
                break
        if not progressed:
            break

    if len(selected) < k:
        selected_ids = {case.id for case in selected}
        for case in fallback_cases:
            if case.id in selected_ids:
                continue
            selected.append(case)
            if len(selected) >= k:
                break

    return selected[:k]


def _expected_output_json(case: RuleKilnCase) -> str:
    if isinstance(case.expected, dict):
        return json.dumps(case.expected, ensure_ascii=False, sort_keys=True)
    if isinstance(case.expected, str):
        return json.dumps({"label": case.expected}, ensure_ascii=False, sort_keys=True)
    return json.dumps({"label": ""}, ensure_ascii=False, sort_keys=True)


def render_few_shot_examples(examples: list[RuleKilnCase]) -> str:
    """Render few-shot examples as deterministic markdown sections."""
    if not examples:
        return ""

    blocks: list[str] = []
    for index, case in enumerate(examples, start=1):
        input_json = json.dumps(case.input, ensure_ascii=False, sort_keys=True, indent=2)
        output_json = _expected_output_json(case)
        blocks.append(
            "\n".join(
                [
                    f"### Example {index}",
                    "Input:",
                    "```json",
                    input_json,
                    "```",
                    "Output:",
                    "```json",
                    output_json,
                    "```",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_few_shot_prompt_with_budget(
    *,
    baseline_prompt: str,
    examples: list[RuleKilnCase],
    max_prompt_tokens: int,
) -> tuple[str, int, list[RuleKilnCase]]:
    """Build few-shot prompt and clip examples deterministically to token budget."""
    if not examples:
        tokens = count_tokens_approx(baseline_prompt)
        return baseline_prompt, tokens, []

    selected_examples = list(examples)
    while selected_examples:
        examples_text = render_few_shot_examples(selected_examples)
        prompt = (
            baseline_prompt
            + "\n\n# Few-Shot Examples\n\nUse these examples as guidance.\n\n"
            + examples_text
        )
        token_count = count_tokens_approx(prompt)
        if token_count <= max_prompt_tokens:
            return prompt, token_count, selected_examples
        selected_examples = selected_examples[:-1]

    baseline_tokens = count_tokens_approx(baseline_prompt)
    return baseline_prompt, baseline_tokens, []


def resolve_distance_metric(distance_metric: str | None) -> str:
    """Resolve configured distance metric with safe fallback."""
    if distance_metric is None:
        return "cosine"
    metric = distance_metric.strip().lower()
    if metric in {"cosine", "euclidean"}:
        return metric
    return "cosine"


def _distance(vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(vec_a - vec_b))

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cosine_similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return float(max(0.0, min(2.0, 1.0 - cosine_similarity)))


def predict_with_centroids(
    *,
    train_embeddings: list[list[float]],
    train_labels: list[str],
    eval_embeddings: list[list[float]],
    metric: str,
) -> list[str]:
    """Predict labels using nearest-centroid classification."""
    if not train_embeddings or not train_labels:
        return ["" for _ in eval_embeddings]

    grouped_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for embedding, label in zip(train_embeddings, train_labels, strict=True):
        grouped_vectors[label].append(np.asarray(embedding, dtype=np.float64))

    centroids: dict[str, np.ndarray] = {}
    for label, vectors in grouped_vectors.items():
        stacked = np.vstack(vectors)
        centroids[label] = stacked.mean(axis=0)

    label_order = sorted(centroids.keys())
    predictions: list[str] = []
    for embedding in eval_embeddings:
        query = np.asarray(embedding, dtype=np.float64)
        ranked = sorted(
            ((_distance(query, centroids[label], metric), label) for label in label_order),
            key=lambda item: (item[0], item[1]),
        )
        predictions.append(ranked[0][1] if ranked else "")
    return predictions


def predict_with_knn(
    *,
    train_embeddings: list[list[float]],
    train_labels: list[str],
    eval_embeddings: list[list[float]],
    metric: str,
    k: int,
    train_ids: list[str] | None = None,
) -> list[str]:
    """Predict labels with deterministic k-NN tie-breaking."""
    if not train_embeddings or not train_labels:
        return ["" for _ in eval_embeddings]
    if k <= 0:
        return ["" for _ in eval_embeddings]

    resolved_train_ids = (
        list(train_ids)
        if train_ids is not None and len(train_ids) == len(train_embeddings)
        else [str(index) for index in range(len(train_embeddings))]
    )

    train_vectors = [np.asarray(embedding, dtype=np.float64) for embedding in train_embeddings]

    predictions: list[str] = []
    for embedding in eval_embeddings:
        query = np.asarray(embedding, dtype=np.float64)
        ranked_neighbors = sorted(
            (
                (_distance(query, train_vec, metric), case_id, label)
                for train_vec, case_id, label in zip(
                    train_vectors, resolved_train_ids, train_labels, strict=True
                )
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        top_neighbors = ranked_neighbors[: min(k, len(ranked_neighbors))]
        if not top_neighbors:
            predictions.append("")
            continue

        vote_counts: dict[str, int] = defaultdict(int)
        vote_distance_sum: dict[str, float] = defaultdict(float)
        for distance, _case_id, label in top_neighbors:
            vote_counts[label] += 1
            vote_distance_sum[label] += distance

        best_count = max(vote_counts.values())
        tied_labels = [label for label, count in vote_counts.items() if count == best_count]
        tied_labels.sort(
            key=lambda label: (
                vote_distance_sum[label] / vote_counts[label],
                label,
            )
        )
        predictions.append(tied_labels[0])

    return predictions


def select_retrieval_examples(
    *,
    query_embedding: list[float],
    train_embeddings: list[list[float]],
    train_cases: list[RuleKilnCase],
    metric: str,
    k: int,
    exclude_case_id: str | None = None,
) -> list[RuleKilnCase]:
    """Return top-k nearest training examples with deterministic tie-breaking."""
    query = np.asarray(query_embedding, dtype=np.float64)
    candidates: list[tuple[float, str, RuleKilnCase]] = []
    for embedding, case in zip(train_embeddings, train_cases, strict=True):
        if exclude_case_id is not None and case.id == exclude_case_id:
            continue
        candidate = np.asarray(embedding, dtype=np.float64)
        distance = _distance(query, candidate, metric)
        candidates.append((distance, case.id, case))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [case for _distance_value, _case_id, case in candidates[:k]]
