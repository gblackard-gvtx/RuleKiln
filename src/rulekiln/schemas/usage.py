"""Model usage, cost, and tracking schemas for provider call observability."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ModelUsage(BaseModel):
    """Token usage from a single model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # OpenAI naming aliases (populated for OpenAI-style providers)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    estimated: bool = False


class ModelCallCost(BaseModel):
    """Estimated USD cost for a single model call."""

    input_cost_usd: Decimal = Decimal("0")
    output_cost_usd: Decimal = Decimal("0")
    total_cost_usd: Decimal = Decimal("0")
    pricing_source: str | None = None
    estimated: bool = True


class ModelCallContext(BaseModel):
    """Immutable context for a single tracked provider call."""

    job_id: str
    stage: str
    role: Literal["teacher", "student", "embedding", "judge"]

    provider_profile: str
    provider: str
    model: str

    student_id: str | None = None
    strategy: str | None = None
    case_id: str | None = None

    prompt_hash: str | None = None
    input_hash: str | None = None


class ModelCallRecord(BaseModel):
    """Full record of a single tracked model call."""

    id: UUID = Field(default_factory=lambda: __import__("uuid").uuid4())
    job_id: str

    stage: str
    role: Literal["teacher", "student", "embedding", "judge"]

    provider_profile: str
    provider: str
    model: str

    student_id: str | None = None
    strategy: str | None = None
    case_id: str | None = None

    usage: ModelUsage
    cost: ModelCallCost

    latency_ms: int | None = None
    status: Literal["success", "failed"]
    error_type: str | None = None


class ChatCompletionResult(BaseModel):
    """Structured result returned by a chat provider call."""

    content: str = ""
    parsed: BaseModel | None = None
    usage: ModelUsage | None = None
    raw_model: str | None = None
    provider_response_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class EmbeddingResult(BaseModel):
    """Result returned by an embedding provider call."""

    embeddings: list[list[float]]
    usage: ModelUsage | None = None
    raw_model: str | None = None
    provider_response_id: str | None = None
