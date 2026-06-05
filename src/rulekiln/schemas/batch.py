"""Batch API domain schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from rulekiln.schemas.usage import ChatCompletionResult, ModelUsage


class BatchItem(BaseModel):
    """A single item in a batch submission."""

    custom_id: str
    system_prompt: str
    user_prompt: str
    output_schema_json: dict[str, object]
    output_schema_class_name: str


class BatchPollStatus(BaseModel):
    """Current status of an in-flight batch."""

    batch_id: str
    provider: str
    processing: bool
    succeeded_count: int
    errored_count: int
    total_count: int
    estimated_completion_at: datetime | None = None


class BatchItemResult(BaseModel):
    """Result for a single item in a collected batch."""

    custom_id: str
    status: Literal["succeeded", "errored", "expired"]
    result: ChatCompletionResult | None = None
    error_message: str | None = None
    usage: ModelUsage | None = None

    model_config = {"arbitrary_types_allowed": True}


class BatchResult(BaseModel):
    """Aggregate result for a completed batch."""

    batch_id: str
    provider: str
    status: Literal["completed", "partial", "failed", "expired"]
    items: list[BatchItemResult]
    succeeded_count: int
    errored_count: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Decimal

    model_config = {"arbitrary_types_allowed": True}
