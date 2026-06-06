"""Deterministic seed generation for evaluation bootstrap sampling."""

from __future__ import annotations

from hashlib import sha256


def _evaluation_bootstrap_seed(
    *,
    job_id: str,
    strategy: str,
    split: str,
    seed_offset: int,
    case_id: str | None = None,
) -> int:
    raw_seed = f"{job_id}|{strategy}|{split}|{seed_offset}|{case_id or ''}"
    digest = sha256(raw_seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
