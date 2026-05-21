"""AWS Bedrock chat provider adapter using pydantic-ai."""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)


class BedrockChatClient(ChatModelClient):
    """Chat adapter for AWS Bedrock via pydantic-ai."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> BaseModel:
        if config.region is None:
            raise ProviderNotConfiguredError("bedrock", "region is required for bedrock provider.")

        model = BedrockConverseModel(config.model, region_name=config.region)
        agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType]
            model,
            result_type=output_schema,  # pyright: ignore[reportArgumentType]
            system_prompt=system_prompt,
        )
        result = await agent.run(user_prompt)
        return result.data  # pyright: ignore[reportReturnType]
