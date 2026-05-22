"""OpenAI chat provider adapter using pydantic-ai."""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)


class OpenAIChatClient(ChatModelClient):
    """Chat adapter for OpenAI models via pydantic-ai."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> BaseModel:
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )

        model = OpenAIModel(config.model, api_key=config.api_key)
        agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType]
            model,
            result_type=output_schema,  # pyright: ignore[reportArgumentType]
            system_prompt=system_prompt,
        )
        result = await agent.run(user_prompt)
        return result.data  # pyright: ignore[reportReturnType]
