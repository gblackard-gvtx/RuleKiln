"""Approved clustering-config registry (Workstream I).

The approved/promoted configuration for a strategy is defined by a checked-in registry
under ``rulekiln/config/approved_strategy_configs/`` rather than inferred from the current
``task.yaml``. Promotion compares a job's runtime clustering config against this registry
by canonical hash, so a run that used a different config (e.g. HDBSCAN ``2/null`` instead
of the promoted ``5/1``) is deterministically detected as not-yet-approved.
"""

from __future__ import annotations

import json
from pathlib import Path

from rulekiln.artifacts.canonical_hash import canonical_hash
from rulekiln.schemas.promotion import ApprovedClusteringConfig, ApprovedStrategyConfig

_APPROVED_DIR = Path(__file__).parent.parent / "config" / "approved_strategy_configs"


def approved_config_hash(config: ApprovedClusteringConfig) -> str:
    """Canonical hash of an approved clustering config's normalized fields."""
    return canonical_hash(config)


def load_approved_config(strategy: str) -> ApprovedStrategyConfig | None:
    """Load the approved config for a strategy, or ``None`` if none is registered.

    The stored ``config_hash`` is overwritten with the freshly computed canonical hash so
    the comparison is always authoritative under the current canonicalization version.
    """
    path = _APPROVED_DIR / f"{strategy}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    approved = ApprovedStrategyConfig.model_validate(data)
    approved.config_hash = approved_config_hash(approved.config)
    return approved
