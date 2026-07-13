"""OpenAI-compatible chat provider (any base_url, e.g. Ollama, Together AI)."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_chat_call
from rulekiln.schemas.usage import ChatCompletionResult

# Use response_format=json_schema (native) instead of tool-calling.
# Small local models (e.g. Gemma 4B) follow the system-prompt's JSON instruction
# and return plain JSON text; they do not reliably produce tool-call responses.
# Native mode uses llama.cpp grammar-constrained generation to enforce the output
# schema, so the model wraps its answer inside the expected field automatically.
_LOCAL_MODEL_PROFILE = OpenAIModelProfile(
    default_structured_output_mode="native",
    supports_json_schema_output=True,
)


class OpenAICompatibleChatClient(ChatModelClient):
    """Chat adapter for OpenAI-compatible endpoints (custom base_url)."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        if config.base_url is None:
            raise ProviderNotConfiguredError(
                "openai_compatible", "base_url is required for openai_compatible provider."
            )
        api_key = config.api_key or "dummy"

        async def _call() -> ChatCompletionResult:
            provider = OpenAIProvider(base_url=config.base_url, api_key=api_key)
            model = OpenAIModel(config.model, provider=provider, profile=_LOCAL_MODEL_PROFILE)  # pyright: ignore[reportArgumentType]
            retry_budget = max(0, config.max_retries)
            agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType,reportCallIssue,reportAssignmentType]
                model,
                output_type=output_schema,  # pyright: ignore[reportArgumentType]
                system_prompt=system_prompt,
                retries=retry_budget,
                output_retries=retry_budget,
            )
            result = await agent.run(user_prompt)

            # Extract raw response text for the content fallback in _call_student.
            # In native mode the model returns a JSON string in a TextPart rather than
            # a tool call, so we can surface it for repair logic downstream.
            raw_text = ""
            for msg in result.new_messages():
                if isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        if isinstance(part, TextPart):
                            raw_text += part.content

            pai_usage = result.usage()
            in_tok = pai_usage.input_tokens
            out_tok = pai_usage.output_tokens
            has_usage = in_tok > 0 or out_tok > 0
            usage = (
                build_usage_from_provider(
                    input_tokens=in_tok if has_usage else None,
                    output_tokens=out_tok if has_usage else None,
                    total_tokens=pai_usage.total_tokens if has_usage else None,
                )
                if has_usage
                else None
            )

            parsed: BaseModel = result.output  # pyright: ignore[reportAttributeAccessIssue]
            return ChatCompletionResult(
                content=raw_text,
                parsed=parsed,
                usage=usage,
                raw_model=config.model,
            )

        return await tracked_chat_call(
            call=_call,
            fallback_input_text=system_prompt + user_prompt,
        )
