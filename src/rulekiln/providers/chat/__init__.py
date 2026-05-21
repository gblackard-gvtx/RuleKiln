"""Chat provider factory: maps ProviderConfig to the correct ChatModelClient."""

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig, ProviderNotImplementedError


def get_chat_client(config: ProviderConfig) -> ChatModelClient:
    """Return the ChatModelClient implementation for the given provider config."""
    match config.provider:
        case "fake":
            from rulekiln.providers.chat.fake import FakeChatClient
            return FakeChatClient()
        case "openai":
            from rulekiln.providers.chat.openai_chat import OpenAIChatClient
            return OpenAIChatClient()
        case "openai_compatible":
            from rulekiln.providers.chat.openai_compatible_chat import OpenAICompatibleChatClient
            return OpenAICompatibleChatClient()
        case "bedrock":
            from rulekiln.providers.chat.bedrock_chat import BedrockChatClient
            return BedrockChatClient()
        case "anthropic":
            from rulekiln.providers.chat.anthropic_chat import AnthropicChatClient
            return AnthropicChatClient()
        case "vertex_gemini":
            from rulekiln.providers.chat.stubs import VertexGeminiChatClient
            return VertexGeminiChatClient()
        case "azure_openai":
            from rulekiln.providers.chat.stubs import AzureOpenAIChatClient
            return AzureOpenAIChatClient()
        case "custom":
            from rulekiln.providers.chat.stubs import CustomChatClient
            return CustomChatClient()
        case _:
            raise ProviderNotImplementedError(config.provider)
