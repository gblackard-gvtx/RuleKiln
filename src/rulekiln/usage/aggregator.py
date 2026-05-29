"""Aggregates model call records into a summary for a job."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from rulekiln.schemas.usage import ModelCallRecord


class ModelUsageAggregator:
    """Computes per-role and overall token/cost totals from a list of ModelCallRecord."""

    def aggregate(self, records: list[ModelCallRecord]) -> dict[str, object]:
        """Return a nested dict with total and per-role breakdowns."""
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = Decimal("0")
        has_estimated = False

        by_role: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": Decimal("0"),
                "call_count": 0,
            }
        )

        for rec in records:
            if rec.usage is not None:
                input_t = rec.usage.input_tokens or 0
                output_t = rec.usage.output_tokens or 0
                tok_total = rec.usage.total_tokens or (input_t + output_t)
                if rec.usage.estimated:
                    has_estimated = True
            else:
                input_t = output_t = tok_total = 0
                has_estimated = True

            total_input += input_t
            total_output += output_t
            total_tokens += tok_total

            cost = rec.cost.total_cost_usd if rec.cost else Decimal("0")
            total_cost += cost

            role = rec.role
            bucket = by_role[role]
            bucket["input_tokens"] = int(bucket["input_tokens"]) + input_t  # type: ignore[arg-type]
            bucket["output_tokens"] = int(bucket["output_tokens"]) + output_t  # type: ignore[arg-type]
            bucket["total_tokens"] = int(bucket["total_tokens"]) + tok_total  # type: ignore[arg-type]
            bucket["cost_usd"] = Decimal(str(bucket["cost_usd"])) + cost  # type: ignore[arg-type]
            bucket["call_count"] = int(bucket["call_count"]) + 1  # type: ignore[arg-type]

        # Convert Decimal to float for serialisation
        serialisable_by_role: dict[str, dict[str, object]] = {
            role: {
                "input_tokens": int(v["input_tokens"]),  # type: ignore[arg-type]
                "output_tokens": int(v["output_tokens"]),  # type: ignore[arg-type]
                "total_tokens": int(v["total_tokens"]),  # type: ignore[arg-type]
                "cost_usd": float(v["cost_usd"]),  # type: ignore[arg-type]
                "call_count": int(v["call_count"]),  # type: ignore[arg-type]
            }
            for role, v in by_role.items()
        }

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "estimated_total_cost_usd": float(total_cost),
            "teacher_cost_usd": float(by_role.get("teacher", {}).get("cost_usd", Decimal("0"))),  # type: ignore[arg-type]
            "student_cost_usd": float(by_role.get("student", {}).get("cost_usd", Decimal("0"))),  # type: ignore[arg-type]
            "embedding_cost_usd": float(by_role.get("embedding", {}).get("cost_usd", Decimal("0"))),  # type: ignore[arg-type]
            "judge_cost_usd": float(by_role.get("judge", {}).get("cost_usd", Decimal("0"))),  # type: ignore[arg-type]
            "has_estimated_usage": has_estimated,
            "total_model_calls": len(records),
            "by_role": serialisable_by_role,
        }
