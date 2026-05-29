"""Anthropic chat provider adapter using the anthropic SDK directly."""

import json

import anthropic
from pydantic import BaseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_chat_call
from rulekiln.schemas.usage import ChatCompletionResult


class AnthropicChatClient(ChatModelClient):
    """Chat adapter for Anthropic Claude models.

    Uses the anthropic SDK directly rather than pydantic-ai to avoid
    version-skew issues between pydantic-ai and the anthropic package.
    Structured output is obtained by injecting the JSON schema into the
    system prompt and parsing the model's response with Pydantic.
    """

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        api_key = config.api_key
        if not api_key:
            raise ProviderNotConfiguredError(
                "anthropic",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )

        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            "Respond with a single JSON object that matches this exact schema "
            "(no markdown fences, no extra text):\n"
            f"{schema_json}"
        )

        async def _call() -> ChatCompletionResult:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            message = await client.messages.create(
                model=config.model,
                max_tokens=4096,
                system=augmented_system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text_blocks = [b.text for b in message.content if hasattr(b, "text")]  # pyright: ignore[reportAttributeAccessIssue]
            raw_text = text_blocks[0].strip() if text_blocks else ""
            parsed = output_schema.model_validate_json(raw_text)

            usage_data = getattr(message, "usage", None)
            usage = None
            if usage_data is not None:
                usage = build_usage_from_provider(
                    input_tokens=getattr(usage_data, "input_tokens", None),
                    output_tokens=getattr(usage_data, "output_tokens", None),
                )

            return ChatCompletionResult(
                content=raw_text,
                parsed=parsed,
                usage=usage,
                raw_model=config.model,
                provider_response_id=getattr(message, "id", None),
            )

        return await tracked_chat_call(
            call=_call,
            fallback_input_text=augmented_system + user_prompt,
        )
