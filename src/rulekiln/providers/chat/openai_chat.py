"""OpenAI chat provider adapter using pydantic-ai."""

import os

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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ProviderNotConfiguredError("openai", "OPENAI_API_KEY is not set.")

        model = OpenAIModel(config.model, api_key=api_key)
        agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType]
            model,
            result_type=output_schema,  # pyright: ignore[reportArgumentType]
            system_prompt=system_prompt,
        )
        result = await agent.run(user_prompt)
        return result.data  # pyright: ignore[reportReturnType]
