"""Deterministic baseline predictors for benchmark comparisons."""

from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from rulekiln.benchmarks.schemas import Banking77Example


def predict_majority_label(
    train_examples: list[Banking77Example],
    eval_examples: list[Banking77Example],
) -> list[str]:
    """Predict the most frequent training label for every evaluation example."""
    if not train_examples:
        raise ValueError("train_examples must not be empty.")

    label_counts = Counter(example.label for example in train_examples)
    majority_label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return [majority_label for _ in eval_examples]


def predict_tfidf_linear_svc(
    train_examples: list[Banking77Example],
    eval_examples: list[Banking77Example],
    *,
    seed: int,
) -> list[str]:
    """Train a deterministic TF-IDF + LinearSVC classifier and predict labels."""
    if not train_examples:
        raise ValueError("train_examples must not be empty.")

    if not eval_examples:
        return []

    train_texts = [example.text for example in train_examples]
    train_labels = [example.label for example in train_examples]
    unique_labels = sorted(set(train_labels))

    if len(unique_labels) < 2:
        return predict_majority_label(train_examples, eval_examples)

    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
        LinearSVC(random_state=seed),
    )
    model.fit(train_texts, train_labels)

    eval_texts = [example.text for example in eval_examples]
    predictions = model.predict(eval_texts).tolist()
    return [str(label) for label in predictions]
