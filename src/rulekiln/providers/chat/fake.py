"""Fake chat provider for offline testing and local development."""

from pydantic import BaseModel

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.schemas.usage import ChatCompletionResult


class FakeChatClient(ChatModelClient):
    """Returns deterministic stub responses populated from output_schema defaults."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        # Build a minimal valid instance by constructing from defaults
        parsed = output_schema.model_validate({})
        usage = build_usage_from_provider(input_tokens=0, output_tokens=0, total_tokens=0)
        return ChatCompletionResult(
            content="",
            parsed=parsed,
            usage=usage,
            raw_model=config.model,
        )
