# BANKING77 RuleKiln Benchmark

## Overview

BANKING77 currently uses a legacy Hugging Face dataset script on the main branch. For Python 3.13 and modern `datasets`, this benchmark loader uses the Parquet conversion from `refs/pr/7`.

This directory contains the RuleKiln benchmark setup for **BANKING77**, a public banking intent-classification dataset from PolyAI.

BANKING77 is a good first benchmark for RuleKiln because it is:

- cleanly labeled
- easy to score deterministically
- representative of customer-service routing and intent classification
- small enough to iterate on quickly
- useful for baseline-vs-hardened student-model evaluation

The benchmark asks:

> Can a smaller or cheaper student model classify banking customer intents more reliably after RuleKiln hardens its prompt?

The intended comparison is:

```text
Baseline:
  student model + baseline task prompt

RuleKiln-hardened:
  student model + baseline task prompt + distilled task-policy rules
```

---

## Dataset Source

Dataset:

```text
PolyAI/banking77
```

Hugging Face:

```text
https://huggingface.co/datasets/PolyAI/banking77
```

Paper:

```text
Efficient Intent Detection with Dual Sentence Encoders
Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić
Proceedings of the 2nd Workshop on NLP for Conversational AI, ACL 2020
```

Paper URL:

```text
https://arxiv.org/abs/2003.04807
```

---

## Initial Results (May 2026)

RuleKiln has completed its first full BANKING77 benchmark run using a local Qwen student model.

### Configuration

* Dataset: BANKING77 (PolyAI)
* Task: Intent Classification
* Teacher Model: OpenAI GPT-5.2
* Student Model: Qwen3.5 4B (local llama.cpp deployment)
* Embedding Model: mxbai-embed-large-v1
* Primary Metric: Macro F1
* Rule Generation Strategies:

  * Baseline
  * DBSCAN
  * HDBSCAN

### Results

| Strategy | Macro F1 | Delta vs Baseline |
| -------- | -------: | ----------------: |
| Baseline |   0.1822 |                 - |
| DBSCAN   |   0.2640 |           +0.0818 |
| HDBSCAN  |   0.3308 |           +0.1486 |

Selected strategy:

```text
HDBSCAN
```

### Summary

RuleKiln improved the local Qwen student from:

```text
Macro F1: 0.1822 -> 0.3308
```

This represents:

```text
+0.1486 absolute improvement
+81.6% relative improvement
```

Additional observations:

```text
Malformed output rate: 0.00%
Golden failures: 0
Quality gates: Passed
```

### Cost

Benchmark run cost:

| Category   |  Cost |
| ---------- | ----: |
| Teacher    | $4.97 |
| Judge      | $1.58 |
| Student    | $0.00 |
| Embeddings | $0.00 |
| Total      | $6.55 |

Total usage:

```text
9.5M tokens
3,287 model calls
```

### Interpretation

These results demonstrate the core RuleKiln thesis:

> Use a stronger teacher model during development to extract task-specific rules, then deploy a cheaper or local student model using a hardened prompt.

The student model was evaluated against a baseline prompt and multiple distilled prompt strategies. HDBSCAN produced the best result and was automatically selected after passing all quality gates.

These results should be considered an early benchmark rather than a final performance claim. Additional datasets, held-out test splits, and larger model comparisons are planned.
---

## License

The Hugging Face dataset listing identifies BANKING77 as licensed under:

```text
Creative Commons Attribution 4.0 International
CC BY 4.0
```

Practical implication:

```text
The dataset can generally be used, shared, and adapted, including for commercial purposes, as long as appropriate attribution is provided.
```

This repository should avoid committing the full dataset by default. Prefer loader and conversion scripts so users can fetch the dataset directly from Hugging Face.

---

## Attribution

Suggested attribution:

```text
BANKING77 is licensed under CC BY 4.0 and was introduced in “Efficient Intent Detection with Dual Sentence Encoders” by Casanueva et al. The dataset is available from PolyAI on Hugging Face.
```

BibTeX:

```bibtex
@inproceedings{casanueva-etal-2020-efficient,
  title = "Efficient Intent Detection with Dual Sentence Encoders",
  author = "Casanueva, Iñigo and Temčinas, Tadas and Gerz, Daniela and Henderson, Matthew and Vulić, Ivan",
  booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
  year = "2020",
  publisher = "Association for Computational Linguistics"
}
```

---
---
