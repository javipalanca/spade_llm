"""Tests for base LLM provider class."""

from abc import ABC

import pytest

from spade_llm.context import ContextManager
from spade_llm.providers.base_provider import BaseLLMProvider


class TestBaseLLMProvider:
    """Test the base LLM provider abstract class."""

    def test_base_provider_is_abstract(self):
        """Test that BaseLLMProvider is an abstract class."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_base_provider_inheritance(self):
        """Test that BaseLLMProvider inherits from ABC."""
        assert issubclass(BaseLLMProvider, ABC)

    def test_abstract_method_exists(self):
        """Test that get_llm_response is an abstract method."""
        assert hasattr(BaseLLMProvider, "get_llm_response")
        assert getattr(BaseLLMProvider.get_llm_response, "__isabstractmethod__", False)


class ConcreteLLMProvider(BaseLLMProvider):
    """Concrete implementation for testing legacy methods."""

    def __init__(self, mock_response=None, mock_tool_calls=None):
        super().__init__()
        self.mock_response = mock_response or "Mock response"
        self.mock_tool_calls = mock_tool_calls or []
        self.call_count = 0

    async def get_llm_response(self, context, tools=None):
        """Mock implementation of get_llm_response."""
        self.call_count += 1
        return {"text": self.mock_response if not self.mock_tool_calls else None, "tool_calls": self.mock_tool_calls}


class TestLegacyMethods:
    """Test the legacy methods that delegate to get_llm_response."""

    @pytest.mark.asyncio
    async def test_get_response_with_text(self, context_manager):
        """Test get_response returns text when available."""
        provider = ConcreteLLMProvider(mock_response="Test response")

        result = await provider.get_response(context_manager)

        assert result == "Test response"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_get_response_with_tools(self, context_manager):
        """Test get_response returns None when tool calls are present."""
        tool_calls = [{"id": "call_1", "name": "test_tool", "arguments": {}}]
        provider = ConcreteLLMProvider(mock_tool_calls=tool_calls)

        result = await provider.get_response(context_manager)

        assert result is None
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_get_tool_calls_with_tools(self, context_manager):
        """Test get_tool_calls returns tool calls when available."""
        tool_calls = [
            {"id": "call_1", "name": "test_tool", "arguments": {"param": "value"}},
            {"id": "call_2", "name": "another_tool", "arguments": {}},
        ]
        provider = ConcreteLLMProvider(mock_tool_calls=tool_calls)

        result = await provider.get_tool_calls(context_manager)

        assert result == tool_calls
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_get_tool_calls_no_tools(self, context_manager):
        """Test get_tool_calls returns empty list when no tools."""
        provider = ConcreteLLMProvider(mock_response="Just text")

        result = await provider.get_tool_calls(context_manager)

        assert result == []
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_legacy_methods_pass_tools(self, context_manager, mock_simple_tool):
        """Test that legacy methods pass tools parameter correctly."""

        class ToolTrackingProvider(ConcreteLLMProvider):
            def __init__(self):
                super().__init__()
                self.received_tools = None

            async def get_llm_response(self, context, tools=None):
                self.received_tools = tools
                return await super().get_llm_response(context, tools)

        provider = ToolTrackingProvider()
        tools = [mock_simple_tool]

        # Test get_response passes tools
        await provider.get_response(context_manager, tools)
        assert provider.received_tools == tools

        # Test get_tool_calls passes tools
        await provider.get_tool_calls(context_manager, tools)
        assert provider.received_tools == tools


class TestProviderInterface:
    """Test the provider interface requirements."""

    def test_concrete_provider_must_implement_get_llm_response(self):
        """Test that concrete providers must implement get_llm_response."""

        class IncompleteProvider(BaseLLMProvider):
            pass  # Missing get_llm_response implementation

        # Should not be able to instantiate without implementing abstract method
        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_concrete_provider_can_be_instantiated(self):
        """Test that properly implemented concrete provider can be instantiated."""
        provider = ConcreteLLMProvider()

        assert isinstance(provider, BaseLLMProvider)
        assert hasattr(provider, "get_llm_response")
        assert callable(provider.get_llm_response)

    @pytest.mark.asyncio
    async def test_get_llm_response_signature(self, context_manager):
        """Test that get_llm_response has correct signature."""
        provider = ConcreteLLMProvider()

        # Should be able to call with context only
        result = await provider.get_llm_response(context_manager)
        assert isinstance(result, dict)
        assert "text" in result
        assert "tool_calls" in result

        # Should be able to call with context and tools
        result = await provider.get_llm_response(context_manager, tools=[])
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_llm_response_return_format(self, context_manager):
        """Test that get_llm_response returns correct format."""
        provider = ConcreteLLMProvider(
            mock_response="Test response", mock_tool_calls=[{"id": "call_1", "name": "tool", "arguments": {}}]
        )

        # Test with text response
        provider.mock_tool_calls = []
        result = await provider.get_llm_response(context_manager)

        assert isinstance(result, dict)
        assert "text" in result
        assert "tool_calls" in result
        assert result["text"] == "Test response"
        assert result["tool_calls"] == []

        # Test with tool calls
        provider.mock_tool_calls = [{"id": "call_1", "name": "tool", "arguments": {}}]
        provider.mock_response = None
        result = await provider.get_llm_response(context_manager)

        assert result["text"] is None
        assert len(result["tool_calls"]) == 1


class TestProviderErrorHandling:
    """Test error handling in provider implementations."""

    @pytest.mark.asyncio
    async def test_provider_can_raise_exceptions(self, context_manager):
        """Test that providers can raise exceptions."""

        class ErrorProvider(BaseLLMProvider):
            async def get_llm_response(self, context, tools=None):
                raise ValueError("Provider error")

        provider = ErrorProvider()

        with pytest.raises(ValueError, match="Provider error"):
            await provider.get_llm_response(context_manager)

    @pytest.mark.asyncio
    async def test_legacy_methods_propagate_exceptions(self, context_manager):
        """Test that legacy methods propagate exceptions from get_llm_response."""

        class ErrorProvider(BaseLLMProvider):
            async def get_llm_response(self, context, tools=None):
                raise RuntimeError("LLM error")

        provider = ErrorProvider()

        # get_response should propagate exception
        with pytest.raises(RuntimeError, match="LLM error"):
            await provider.get_response(context_manager)

        # get_tool_calls should propagate exception
        with pytest.raises(RuntimeError, match="LLM error"):
            await provider.get_tool_calls(context_manager)


class TestProviderWithTools:
    """Test provider behavior with tools."""

    @pytest.mark.asyncio
    async def test_provider_receives_tools_parameter(self, context_manager, mock_simple_tool, mock_async_tool):
        """Test that provider receives tools parameter correctly."""

        class ToolInspectingProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__()
                self.last_tools = None

            async def get_llm_response(self, context, tools=None):
                self.last_tools = tools
                return {"text": "response", "tool_calls": []}

        provider = ToolInspectingProvider()
        tools = [mock_simple_tool, mock_async_tool]

        await provider.get_llm_response(context_manager, tools)

        assert provider.last_tools == tools
        assert len(provider.last_tools) == 2


class TestProviderContextHandling:
    """Test how providers handle context."""

    @pytest.mark.asyncio
    async def test_provider_receives_context(self, context_manager):
        """Test that provider receives context parameter."""

        class ContextInspectingProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__()
                self.last_context = None

            async def get_llm_response(self, context, tools=None):
                self.last_context = context
                return {"text": "response", "tool_calls": []}

        provider = ContextInspectingProvider()

        await provider.get_llm_response(context_manager)

        assert provider.last_context == context_manager
        assert isinstance(provider.last_context, ContextManager)

    @pytest.mark.asyncio
    async def test_provider_can_access_context_methods(self, context_manager):
        """Test that provider can access context methods."""

        class ContextUsingProvider(BaseLLMProvider):
            async def get_llm_response(self, context, tools=None):
                # Provider should be able to call context methods
                prompt = context.get_prompt()
                conversations = context.get_active_conversations()

                return {
                    "text": f"Prompt length: {len(prompt)}, Active conversations: {len(conversations)}",
                    "tool_calls": [],
                }

        provider = ContextUsingProvider()

        # Add some data to context
        context_manager.add_assistant_message("Test message", "test_conversation")

        result = await provider.get_llm_response(context_manager)

        assert "Prompt length:" in result["text"]
        assert "Active conversations:" in result["text"]
