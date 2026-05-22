"""Rule pruning service: selects rules within budget constraints before prompt compilation."""

from __future__ import annotations

from typing import Literal

from rulekiln.pipeline.prompt_compiler import count_tokens_approx
from rulekiln.schemas.pipeline import SynthesizedRuleSchema

PruningReason = Literal[
    "unresolved_conflict",
    "below_min_support",
    "max_rules_exceeded",
    "prompt_token_budget_exceeded",
    "duplicate_or_subsumed",
]


class PruningRecord:
    """Records the pruning decision for a single rule."""

    def __init__(self, rule: SynthesizedRuleSchema, reason: PruningReason) -> None:
        self.rule = rule
        self.reason = reason


class PruningResult:
    """Output of the rule pruning service."""

    def __init__(
        self,
        selected: list[SynthesizedRuleSchema],
        pruned: list[PruningRecord],
    ) -> None:
        self.selected = selected
        self.pruned = pruned

    @property
    def pruned_count(self) -> int:
        return len(self.pruned)

    @property
    def selected_count(self) -> int:
        return len(self.selected)

    def pruning_report(self) -> dict[str, int | list[dict[str, str]]]:
        reason_counts: dict[str, int] = {}
        pruned_entries: list[dict[str, str]] = []
        for record in self.pruned:
            reason_counts[record.reason] = reason_counts.get(record.reason, 0) + 1
            pruned_entries.append({"rule_id": record.rule.id, "topic": record.rule.topic, "reason": record.reason})
        return {
            "selected_count": self.selected_count,
            "pruned_count": self.pruned_count,
            "reason_counts": reason_counts,  # type: ignore[dict-item]
            "pruned_rules": pruned_entries,  # type: ignore[dict-item]
        }


def prune_rules(
    rules: list[SynthesizedRuleSchema],
    *,
    max_rules: int = 40,
    max_prompt_tokens: int = 8000,
    min_rule_support_count: int = 2,
    preserve_golden_rules: bool = True,
    golden_case_ids: set[str] | None = None,
) -> PruningResult:
    """Apply the full pruning pipeline to a list of synthesized rules.

    Pruning order (spec §10.4):
    1. Remove unresolved-conflict rules.
    2. Preserve rules backed by golden cases (if preserve_golden_rules).
    3. Remove rules below min_rule_support_count, unless golden-backed.
    4. Sort by priority asc, support_count desc, support_ratio desc.
    5. Cap at max_rules.
    6. Cap at max_prompt_tokens token budget.
    """
    selected: list[SynthesizedRuleSchema] = []
    pruned: list[PruningRecord] = []

    golden_ids = golden_case_ids or set()

    for rule in rules:
        # 1. Remove unresolved conflicts (has_conflicts and not overridden by the
        #    conflict review keeping it; if has_conflicts is True at this point the
        #    conflict was not resolved to "keep" or "modify").
        if rule.has_conflicts:
            pruned.append(PruningRecord(rule, "unresolved_conflict"))
            continue

        # Tag golden-case-backed status if not already set
        is_golden = rule.golden_case_backed or bool(
            golden_ids and set(rule.source_case_ids) & golden_ids
        )

        # 2/3. Remove below-min-support unless golden-backed
        if rule.support_count < min_rule_support_count:
            if not (preserve_golden_rules and is_golden):
                pruned.append(PruningRecord(rule, "below_min_support"))
                continue

        selected.append(rule)

    # 4. Sort: priority asc (lower = higher priority), support_count desc, support_ratio desc
    selected.sort(key=lambda r: (r.priority, -r.support_count, -r.support_ratio))

    # 5. Cap at max_rules
    if len(selected) > max_rules:
        over = selected[max_rules:]
        for rule in over:
            pruned.append(PruningRecord(rule, "max_rules_exceeded"))
        selected = selected[:max_rules]

    # 6. Cap at prompt token budget
    running_tokens = 0
    within_budget: list[SynthesizedRuleSchema] = []
    for rule in selected:
        # Use stored estimate if available, otherwise compute it
        token_count = rule.estimated_token_count or count_tokens_approx(
            f"{rule.topic} {' '.join(rule.applies_when)}"
        )
        if running_tokens + token_count > max_prompt_tokens:
            pruned.append(PruningRecord(rule, "prompt_token_budget_exceeded"))
        else:
            running_tokens += token_count
            within_budget.append(rule)

    return PruningResult(selected=within_budget, pruned=pruned)
