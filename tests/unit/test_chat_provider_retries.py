"""Regression tests for provider-configured pydantic-ai retry wiring."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rulekiln.providers.chat.bedrock_chat import BedrockChatClient
from rulekiln.providers.chat.openai_chat import OpenAIChatClient
from rulekiln.providers.chat.openai_compatible_chat import OpenAICompatibleChatClient
from rulekiln.providers.contracts import ProviderConfig


class _DummyOutput(BaseModel):
    """Simple structured output model used by retry wiring tests."""

    ok: bool = True


class _FakeUsage:
    """Minimal pydantic-ai usage object shape expected by adapters."""

    input_tokens: int = 11
    output_tokens: int = 7
    total_tokens: int = 18


class _FakeRunResult:
    """Minimal pydantic-ai run result shape expected by adapters."""

    output: BaseModel = _DummyOutput()

    def usage(self) -> _FakeUsage:
        return _FakeUsage()


@pytest.mark.asyncio
async def test_openai_chat_uses_configured_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI adapter should pass ProviderConfig.max_retries into Agent(...)."""
    import rulekiln.providers.chat.openai_chat as module

    captured: dict[str, int] = {}

    class _FakeAgent:
        def __init__(
            self,
            _model: object,
            *,
            output_type: type[BaseModel],
            system_prompt: str,
            retries: int,
            output_retries: int,
        ) -> None:
            assert output_type is _DummyOutput
            assert system_prompt == "sys"
            captured["retries"] = retries
            captured["output_retries"] = output_retries

        async def run(self, _user_prompt: str) -> _FakeRunResult:
            return _FakeRunResult()

    class _FakeProvider:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _FakeModel:
        def __init__(self, _model_name: str, *, provider: object) -> None:
            assert provider is not None

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "OpenAIProvider", _FakeProvider)
    monkeypatch.setattr(module, "OpenAIModel", _FakeModel)

    client = OpenAIChatClient()
    result = await client.complete_structured(
        system_prompt="sys",
        user_prompt="user",
        output_schema=_DummyOutput,
        config=ProviderConfig(
            profile_name="p",
            provider="openai",
            model="gpt-test",
            api_key="test-key",
            max_retries=5,
        ),
    )

    assert captured == {"retries": 5, "output_retries": 5}
    assert isinstance(result.parsed, _DummyOutput)


@pytest.mark.asyncio
async def test_openai_compatible_chat_uses_configured_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible adapter should pass ProviderConfig.max_retries into Agent(...)."""
    import rulekiln.providers.chat.openai_compatible_chat as module

    captured: dict[str, int] = {}

    class _FakeAgent:
        def __init__(
            self,
            _model: object,
            *,
            output_type: type[BaseModel],
            system_prompt: str,
            retries: int,
            output_retries: int,
        ) -> None:
            assert output_type is _DummyOutput
            assert system_prompt == "sys"
            captured["retries"] = retries
            captured["output_retries"] = output_retries

        async def run(self, _user_prompt: str) -> _FakeRunResult:
            return _FakeRunResult()

    class _FakeProvider:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _FakeModel:
        def __init__(self, _model_name: str, *, provider: object) -> None:
            assert provider is not None

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "OpenAIProvider", _FakeProvider)
    monkeypatch.setattr(module, "OpenAIModel", _FakeModel)

    client = OpenAICompatibleChatClient()
    result = await client.complete_structured(
        system_prompt="sys",
        user_prompt="user",
        output_schema=_DummyOutput,
        config=ProviderConfig(
            profile_name="p",
            provider="openai_compatible",
            model="compat-test",
            base_url="http://localhost:9999/v1",
            api_key="test-key",
            max_retries=4,
        ),
    )

    assert captured == {"retries": 4, "output_retries": 4}
    assert isinstance(result.parsed, _DummyOutput)


@pytest.mark.asyncio
async def test_bedrock_chat_uses_configured_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bedrock adapter should pass ProviderConfig.max_retries into Agent(...)."""
    import rulekiln.providers.chat.bedrock_chat as module

    captured: dict[str, int] = {}

    class _FakeAgent:
        def __init__(
            self,
            _model: object,
            *,
            output_type: type[BaseModel],
            system_prompt: str,
            retries: int,
            output_retries: int,
        ) -> None:
            assert output_type is _DummyOutput
            assert system_prompt == "sys"
            captured["retries"] = retries
            captured["output_retries"] = output_retries

        async def run(self, _user_prompt: str) -> _FakeRunResult:
            return _FakeRunResult()

    class _FakeModel:
        def __init__(self, _model_name: str, *, region_name: str) -> None:
            assert region_name == "us-east-1"

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "BedrockConverseModel", _FakeModel)

    client = BedrockChatClient()
    result = await client.complete_structured(
        system_prompt="sys",
        user_prompt="user",
        output_schema=_DummyOutput,
        config=ProviderConfig(
            profile_name="p",
            provider="bedrock",
            model="bedrock-test",
            region="us-east-1",
            max_retries=6,
        ),
    )

    assert captured == {"retries": 6, "output_retries": 6}
    assert isinstance(result.parsed, _DummyOutput)
