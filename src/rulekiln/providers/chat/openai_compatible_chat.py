"""OpenAI-compatible chat provider (any base_url, e.g. Ollama, Together AI)."""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_chat_call
from rulekiln.schemas.usage import ChatCompletionResult


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
        # api_key may be a dummy value for local servers that require the header
        api_key = config.api_key or "dummy"

        async def _call() -> ChatCompletionResult:
            provider = OpenAIProvider(base_url=config.base_url, api_key=api_key)
            model = OpenAIModel(config.model, provider=provider)
            retry_budget = max(0, config.max_retries)
            agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType]
                model,
                output_type=output_schema,  # pyright: ignore[reportArgumentType]
                system_prompt=system_prompt,
                retries=retry_budget,
                output_retries=retry_budget,
            )
            result = await agent.run(user_prompt)
            pai_usage = result.usage()
            # Local servers may return zero tokens; treat as estimated if so
            input_tokens = pai_usage.input_tokens
            output_tokens = pai_usage.output_tokens
            has_usage = input_tokens > 0 or output_tokens > 0
            usage = build_usage_from_provider(
                input_tokens=input_tokens if has_usage else None,
                output_tokens=output_tokens if has_usage else None,
                total_tokens=pai_usage.total_tokens if has_usage else None,
            ) if has_usage else None

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
