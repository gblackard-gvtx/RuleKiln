"""Anthropic chat provider adapter using the anthropic SDK directly."""

import json
import os

import anthropic
from pydantic import BaseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)


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
    ) -> BaseModel:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ProviderNotConfiguredError("anthropic", "ANTHROPIC_API_KEY is not set.")

        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            "Respond with a single JSON object that matches this exact schema "
            "(no markdown fences, no extra text):\n"
            f"{schema_json}"
        )

        client = anthropic.AsyncAnthropic(api_key=api_key)
        message = await client.messages.create(
            model=config.model,
            max_tokens=4096,
            system=augmented_system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text_blocks = [b.text for b in message.content if hasattr(b, "text")]  # pyright: ignore[reportAttributeAccessIssue]
        raw_text = text_blocks[0].strip() if text_blocks else ""
        return output_schema.model_validate_json(raw_text)
