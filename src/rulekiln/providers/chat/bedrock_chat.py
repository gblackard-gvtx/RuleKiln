"""AWS Bedrock chat provider adapter using pydantic-ai."""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_chat_call
from rulekiln.schemas.usage import ChatCompletionResult


class BedrockChatClient(ChatModelClient):
    """Chat adapter for AWS Bedrock via pydantic-ai."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        if config.region is None:
            raise ProviderNotConfiguredError("bedrock", "region is required for bedrock provider.")

        async def _call() -> ChatCompletionResult:
            model = BedrockConverseModel(config.model, region_name=config.region)
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
