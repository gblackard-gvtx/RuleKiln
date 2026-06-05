"""OpenAI chat provider adapter using pydantic-ai (per-call) and the OpenAI SDK (batch)."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rulekiln.providers.batch_schema_registry import get_schema_class
from rulekiln.providers.contracts import (
    BatchChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_chat_call
from rulekiln.schemas.batch import (
    BatchItem,
    BatchItemResult,
    BatchPollStatus,
    BatchResult,
)
from rulekiln.schemas.usage import ChatCompletionResult, ModelUsage

# Batch statuses that indicate the job is still in flight.
_OPENAI_PROCESSING_STATUSES: frozenset[str] = frozenset({"validating", "in_progress", "finalizing"})


def _extract_response_text(response_body: dict[str, object]) -> str | None:
    """Return the first output_text from a /v1/responses response body.

    Traverses ``output[*].content[*]`` looking for the first item whose
    ``type`` is ``"output_text"``, matching the Responses API schema.
    Defensive against missing or malformed keys.
    """
    output = response_body.get("output")
    if not isinstance(output, list):
        return None
    for output_item in output:
        if not isinstance(output_item, dict) or output_item.get("type") != "message":
            continue
        content = output_item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str):
                    return text
    return None


def _parse_usage(response_body: dict[str, object]) -> ModelUsage | None:
    """Extract token counts from a /v1/responses response body."""
    usage = response_body.get("usage")
    if not isinstance(usage, dict):
        return None
    return ModelUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )


class OpenAIChatClient(BatchChatModelClient):
    """Chat adapter for OpenAI models.

    Per-call path uses pydantic-ai for structured output and retries.
    Batch path uses the OpenAI SDK directly against the /v1/responses endpoint.
    """

    # ── Per-call ──────────────────────────────────────────────────────────────

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )

        async def _call() -> ChatCompletionResult:
            provider = OpenAIProvider(api_key=config.api_key)
            model = OpenAIModel(config.model, provider=provider)
            retry_budget = max(0, config.max_retries)
            agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType,reportCallIssue,reportAssignmentType]
                model,
                output_type=output_schema,  # pyright: ignore[reportArgumentType]
                system_prompt=system_prompt,
                retries=retry_budget,
                output_retries=retry_budget,
            )
            result = await agent.run(user_prompt)
            pai_usage = result.usage()
            usage = build_usage_from_provider(
                input_tokens=pai_usage.input_tokens,
                output_tokens=pai_usage.output_tokens,
                total_tokens=pai_usage.total_tokens,
            )
            parsed: BaseModel = result.output  # pyright: ignore[reportAttributeAccessIssue]
            return ChatCompletionResult(
                content="",
                parsed=parsed,
                usage=usage,
                raw_model=config.model,
            )

        return await tracked_chat_call(
            call=_call,
            fallback_input_text=system_prompt + user_prompt,
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def submit_batch(
        self,
        items: list[BatchItem],
        config: ProviderConfig,
    ) -> str:
        """Serialize *items* to JSONL, upload to OpenAI Files, and create a batch job.

        Returns the OpenAI batch ID (``batch_*``).
        """
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )
        import openai

        client = openai.AsyncOpenAI(api_key=config.api_key)

        jsonl_bytes = _build_batch_jsonl(items, config.model)
        uploaded = await client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )

        batch = await client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        return batch.id

    async def poll_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
    ) -> BatchPollStatus:
        """Return the current status of an OpenAI batch without blocking."""
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )
        import openai

        client = openai.AsyncOpenAI(api_key=config.api_key)
        batch = await client.batches.retrieve(batch_id)

        processing = batch.status in _OPENAI_PROCESSING_STATUSES
        rc = batch.request_counts

        estimated_at: datetime | None = None
        if batch.expires_at:
            estimated_at = datetime.fromtimestamp(batch.expires_at, tz=UTC)

        return BatchPollStatus(
            batch_id=batch_id,
            provider="openai",
            processing=processing,
            succeeded_count=rc.completed if rc else 0,
            errored_count=rc.failed if rc else 0,
            total_count=rc.total if rc else 0,
            estimated_completion_at=estimated_at,
        )

    async def collect_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
        *,
        output_schema_class_name: str,
    ) -> BatchResult:
        """Download and parse a completed OpenAI batch.

        Raises ``ValueError`` if the batch has not finished (still processing).
        """
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )
        import openai

        client = openai.AsyncOpenAI(api_key=config.api_key)
        batch = await client.batches.retrieve(batch_id)

        if batch.status in _OPENAI_PROCESSING_STATUSES:
            raise ValueError(f"Batch {batch_id!r} is not yet complete (status={batch.status!r}).")

        schema_cls = get_schema_class(output_schema_class_name)
        items: list[BatchItemResult] = []
        total_input = 0
        total_output = 0

        # Parse successful output
        if batch.output_file_id:
            raw_output = await client.files.content(batch.output_file_id)
            for line in raw_output.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row: dict[str, object] = json.loads(line)
                custom_id = str(row.get("custom_id", ""))
                response = row.get("response")
                if not isinstance(response, dict):
                    items.append(
                        BatchItemResult(
                            custom_id=custom_id,
                            status="errored",
                            error_message="Missing response object in output JSONL",
                        )
                    )
                    continue

                status_code = response.get("status_code")
                body = response.get("body")
                if status_code != 200 or not isinstance(body, dict):
                    items.append(
                        BatchItemResult(
                            custom_id=custom_id,
                            status="errored",
                            error_message=f"Non-200 status_code={status_code}",
                        )
                    )
                    continue

                text = _extract_response_text(body)
                usage = _parse_usage(body)
                if text is None:
                    items.append(
                        BatchItemResult(
                            custom_id=custom_id,
                            status="errored",
                            error_message="No output_text found in response body",
                        )
                    )
                    continue

                try:
                    parsed = schema_cls.model_validate_json(text)
                except Exception as exc:
                    items.append(
                        BatchItemResult(
                            custom_id=custom_id,
                            status="errored",
                            error_message=f"Schema parse error: {exc}",
                        )
                    )
                    continue

                if usage:
                    total_input += usage.input_tokens or 0
                    total_output += usage.output_tokens or 0

                items.append(
                    BatchItemResult(
                        custom_id=custom_id,
                        status="succeeded",
                        result=ChatCompletionResult(
                            content=text,
                            parsed=parsed,
                            usage=usage,
                            raw_model=config.model,
                        ),
                        usage=usage,
                    )
                )

        # Parse error output — items here failed at the provider level
        if batch.error_file_id:
            raw_errors = await client.files.content(batch.error_file_id)
            for line in raw_errors.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                custom_id = str(row.get("custom_id", ""))
                error = row.get("error") or {}
                if isinstance(error, dict):
                    msg = str(error.get("message", "Provider error"))
                else:
                    msg = str(error)
                items.append(
                    BatchItemResult(
                        custom_id=custom_id,
                        status="errored",
                        error_message=msg,
                    )
                )

        # Any custom_id not in the output or error files was expired
        returned_ids = {item.custom_id for item in items}
        if batch.status == "expired":
            # All non-returned items are expired; we can't enumerate them here since
            # we only have results — the caller handles missing markers as expired.
            batch_status: str = "expired"
        elif batch.status == "failed":
            batch_status = "failed"
        elif any(i.status != "succeeded" for i in items):
            batch_status = "partial"
        else:
            batch_status = "completed"

        succeeded = sum(1 for i in items if i.status == "succeeded")
        errored = len(items) - succeeded
        _ = returned_ids  # used above for expired detection; kept for clarity

        return BatchResult(
            batch_id=batch_id,
            provider="openai",
            status=batch_status,  # type: ignore[arg-type]
            items=items,
            succeeded_count=succeeded,
            errored_count=errored,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            estimated_cost_usd=Decimal("0"),  # caller updates via PricingService.calculate_batch
        )


def _build_batch_jsonl(items: list[BatchItem], model: str) -> bytes:
    """Serialize a list of BatchItems to a JSONL byte string for /v1/responses."""
    lines: list[bytes] = []
    for item in items:
        body: dict[str, object] = {
            "model": model,
            "input": [
                {"role": "system", "content": item.system_prompt},
                {"role": "user", "content": item.user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": item.output_schema_class_name,
                    "schema": item.output_schema_json,
                    "strict": True,
                }
            },
        }
        request: dict[str, object] = {
            "custom_id": item.custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        lines.append(json.dumps(request, ensure_ascii=False).encode())
    return b"\n".join(lines)
