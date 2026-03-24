"""Tests for the unified LLM provider implementation using LiteLLM."""

import pytest
from unittest.mock import Mock, patch

from spade_llm.providers.llm_provider import LLMProvider
from spade_llm.context import ContextManager
from spade_llm.tools import LLMTool


class TestLLMProviderInit:
    """Test LLMProvider initialization and configuration."""

    def test_init_with_required_params(self):
        """Test initialization with required model parameter."""
        provider = LLMProvider(model="gpt-4o-mini")

        assert provider.model == "gpt-4o-mini"
        assert provider.api_key is None
        assert provider.base_url is None
        assert provider.temperature == 1.0
        assert provider.timeout == 600.0
        assert provider.max_tokens is None
        assert provider.num_retries == 0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        provider = LLMProvider(
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
            base_url="http://localhost:8000",
            timeout=120.0,
            max_tokens=1000,
            num_retries=3,
        )

        assert provider.model == "gpt-4"
        assert provider.api_key == "test-key"
        assert provider.temperature == 0.5
        assert provider.base_url == "http://localhost:8000"
        assert provider.timeout == 120.0
        assert provider.max_tokens == 1000
        assert provider.num_retries == 3

    def test_init_with_litellm_kwargs(self):
        """Test initialization with extra kwargs passed through to LiteLLM."""
        provider = LLMProvider(
            model="gpt-4o-mini",
            api_key="test-key",
            custom_llm_provider="openai",
        )

        assert provider.kwargs == {"custom_llm_provider": "openai"}

    def test_init_ollama_model(self):
        """Test initialization with Ollama model format."""
        provider = LLMProvider(
            model="ollama/llama3:8b",
            base_url="http://localhost:11434",
        )

        assert provider.model == "ollama/llama3:8b"
        assert provider.base_url == "http://localhost:11434"
        assert provider.api_key is None


class TestBuildCompletionKwargs:
    """Test the _build_completion_kwargs helper method."""

    def test_build_kwargs_basic(self):
        """Test building kwargs with basic parameters."""
        provider = LLMProvider(model="gpt-4o-mini", api_key="test-key")
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {}

        messages = [{"role": "user", "content": "hello"}]
        kwargs = provider._build_completion_kwargs(mock_context, messages)

        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"] == messages
        assert kwargs["temperature"] == 1.0
        assert kwargs["timeout"] == 600.0
        assert kwargs["api_key"] == "test-key"
        assert "tools" not in kwargs

    def test_build_kwargs_with_tools(self):
        """Test building kwargs includes tools when provided."""
        provider = LLMProvider(model="gpt-4o-mini")
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {}

        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        kwargs = provider._build_completion_kwargs(mock_context, messages, tools)

        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"

    def test_build_kwargs_with_base_url(self):
        """Test building kwargs includes api_base when base_url is set."""
        provider = LLMProvider(model="ollama/llama3:8b", base_url="http://localhost:11434")
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {}

        messages = [{"role": "user", "content": "hello"}]
        kwargs = provider._build_completion_kwargs(mock_context, messages)

        assert kwargs["api_base"] == "http://localhost:11434"

    def test_build_kwargs_with_max_tokens(self):
        """Test building kwargs includes max_tokens when set."""
        provider = LLMProvider(model="gpt-4o-mini", max_tokens=1000)
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {}

        messages = [{"role": "user", "content": "hello"}]
        kwargs = provider._build_completion_kwargs(mock_context, messages)

        assert kwargs["max_tokens"] == 1000

    def test_build_kwargs_without_max_tokens(self):
        """Test building kwargs excludes max_tokens when not set."""
        provider = LLMProvider(model="gpt-4o-mini")
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {}

        messages = [{"role": "user", "content": "hello"}]
        kwargs = provider._build_completion_kwargs(mock_context, messages)

        assert "max_tokens" not in kwargs

    def test_build_kwargs_includes_tracing_metadata(self):
        """Test building kwargs includes tracing metadata."""
        provider = LLMProvider(model="gpt-4o-mini")
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {
            "conversation_id": "conv-123",
            "sender_id": "agent1",
            "receiver_id": "agent2",
        }

        messages = [{"role": "user", "content": "hello"}]
        kwargs = provider._build_completion_kwargs(mock_context, messages)

        assert kwargs["metadata"]["session_id"] == "conv-123"
        assert "agent1" in kwargs["metadata"]["tags"]
        assert "agent2" in kwargs["metadata"]["tags"]

    def test_build_kwargs_extra_kwargs_passed(self):
        """Test that extra kwargs from __init__ are included."""
        provider = LLMProvider(model="gpt-4o-mini", custom_llm_provider="openai")
        mock_context = Mock(spec=ContextManager)
        mock_context.get_tracing_metadata.return_value = {}

        messages = [{"role": "user", "content": "hello"}]
        kwargs = provider._build_completion_kwargs(mock_context, messages)

        assert kwargs["custom_llm_provider"] == "openai"


class TestGetLLMResponse:
    """Test the main get_llm_response method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = Mock(spec=ContextManager)
        self.mock_context.get_prompt.return_value = [{"role": "user", "content": "test"}]
        self.mock_context.get_tracing_metadata.return_value = {}

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_text_only(self, mock_acompletion):
        """Test get_llm_response with text response only."""
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context)

        assert result["text"] == "Test response"
        assert result["tool_calls"] == []

        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 1.0

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_with_tools(self, mock_acompletion):
        """Test get_llm_response with tool calls."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        mock_tool = Mock(spec=LLMTool)
        mock_tool.to_openai_tool.return_value = {
            "function": {"name": "test_tool"},
            "type": "function",
        }

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context, tools=[mock_tool])

        assert result["text"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["name"] == "test_tool"
        assert result["tool_calls"][0]["arguments"] == {"param": "value"}

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_with_max_tokens(self, mock_acompletion):
        """Test get_llm_response includes max_tokens when set."""
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini", max_tokens=1000)
        await provider.get_llm_response(self.mock_context)

        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["max_tokens"] == 1000

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_json_decode_error(self, mock_acompletion):
        """Test handling of JSON decode error in tool arguments."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "invalid json{"

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"] == {}

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_api_error(self, mock_acompletion):
        """Test handling of API errors."""
        mock_acompletion.side_effect = Exception("API Error")

        provider = LLMProvider(model="gpt-4o-mini")

        with pytest.raises(Exception, match="API Error"):
            await provider.get_llm_response(self.mock_context)

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_empty_content(self, mock_acompletion):
        """Test handling of empty response content."""
        mock_message = Mock()
        mock_message.content = ""
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context)

        assert result["text"] == ""
        assert result["tool_calls"] == []

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_none_content(self, mock_acompletion):
        """Test handling of None response content (no tool calls either)."""
        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context)

        assert result["text"] == ""
        assert result["tool_calls"] == []

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_multiple_tool_calls(self, mock_acompletion):
        """Test handling of multiple tool calls in a single response."""
        mock_tc1 = Mock()
        mock_tc1.id = "call_1"
        mock_tc1.function.name = "tool_a"
        mock_tc1.function.arguments = '{"x": 1}'

        mock_tc2 = Mock()
        mock_tc2.id = "call_2"
        mock_tc2.function.name = "tool_b"
        mock_tc2.function.arguments = '{"y": 2}'

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc1, mock_tc2]

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "tool_a"
        assert result["tool_calls"][1]["name"] == "tool_b"

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_dict_arguments(self, mock_acompletion):
        """Test handling of tool call arguments already parsed as dict."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = {"param": "value"}

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_llm_response(self.mock_context)

        assert result["tool_calls"][0]["arguments"] == {"param": "value"}

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_with_conversation_id(self, mock_acompletion):
        """Test get_llm_response passes conversation_id to get_prompt."""
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        provider = LLMProvider(model="gpt-4o-mini")
        await provider.get_llm_response(self.mock_context, conversation_id="conv-123")

        self.mock_context.get_prompt.assert_called_once_with("conv-123")

    @patch("spade_llm.providers.llm_provider.litellm.acompletion")
    @pytest.mark.asyncio
    async def test_get_llm_response_ollama_with_tools(self, mock_acompletion):
        """Test get_llm_response with Ollama provider using tools."""
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_acompletion.return_value = mock_response

        mock_tool = Mock(spec=LLMTool)
        mock_tool.to_openai_tool.return_value = {
            "function": {"name": "test_tool"},
            "type": "function",
        }

        provider = LLMProvider(model="ollama/llama3:8b")
        result = await provider.get_llm_response(self.mock_context, tools=[mock_tool])

        call_kwargs = mock_acompletion.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"


class TestLegacyMethods:
    """Test legacy methods that delegate to get_llm_response."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = Mock(spec=ContextManager)

    @patch.object(LLMProvider, "get_llm_response")
    @pytest.mark.asyncio
    async def test_get_response_returns_text(self, mock_get_llm_response):
        """Test get_response returns text from get_llm_response."""
        mock_get_llm_response.return_value = {
            "text": "Test response",
            "tool_calls": [],
        }

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_response(self.mock_context)

        assert result == "Test response"
        mock_get_llm_response.assert_called_once_with(self.mock_context, None)

    @patch.object(LLMProvider, "get_llm_response")
    @pytest.mark.asyncio
    async def test_get_response_with_tools_parameter(self, mock_get_llm_response):
        """Test get_response passes tools parameter."""
        mock_get_llm_response.return_value = {"text": "Response", "tool_calls": []}

        mock_tools = [Mock()]
        provider = LLMProvider(model="gpt-4o-mini")
        await provider.get_response(self.mock_context, tools=mock_tools)

        mock_get_llm_response.assert_called_once_with(self.mock_context, mock_tools)

    @patch.object(LLMProvider, "get_llm_response")
    @pytest.mark.asyncio
    async def test_get_tool_calls_returns_list(self, mock_get_llm_response):
        """Test get_tool_calls returns tool_calls from get_llm_response."""
        expected_calls = [{"id": "call_1", "name": "tool", "arguments": {}}]
        mock_get_llm_response.return_value = {
            "text": None,
            "tool_calls": expected_calls,
        }

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_tool_calls(self.mock_context)

        assert result == expected_calls
        mock_get_llm_response.assert_called_once_with(self.mock_context, None)

    @patch.object(LLMProvider, "get_llm_response")
    @pytest.mark.asyncio
    async def test_get_tool_calls_empty_list(self, mock_get_llm_response):
        """Test get_tool_calls returns empty list when no tool_calls."""
        mock_get_llm_response.return_value = {
            "text": "Just text",
            "tool_calls": [],
        }

        provider = LLMProvider(model="gpt-4o-mini")
        result = await provider.get_tool_calls(self.mock_context)

        assert result == []


class TestIntegration:
    """Integration tests for the LLMProvider."""

    @pytest.mark.asyncio
    async def test_provider_workflow_text_response(self, context_manager):
        """Test complete workflow with text response."""
        with patch("spade_llm.providers.llm_provider.litellm.acompletion") as mock_acompletion:
            mock_message = Mock()
            mock_message.content = "Hello, how can I help you?"
            mock_message.tool_calls = None

            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]
            mock_acompletion.return_value = mock_response

            provider = LLMProvider(model="gpt-4o-mini", api_key="test-key")

            llm_response = await provider.get_llm_response(context_manager)
            text_response = await provider.get_response(context_manager)
            tool_calls = await provider.get_tool_calls(context_manager)

            assert llm_response["text"] == "Hello, how can I help you?"
            assert llm_response["tool_calls"] == []
            assert text_response == "Hello, how can I help you?"
            assert tool_calls == []

    @pytest.mark.asyncio
    async def test_provider_workflow_tool_calls(self, context_manager):
        """Test complete workflow with tool calls."""
        with patch("spade_llm.providers.llm_provider.litellm.acompletion") as mock_acompletion:
            mock_tool_call = Mock()
            mock_tool_call.id = "call_abc123"
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"location": "Paris"}'

            mock_message = Mock()
            mock_message.content = None
            mock_message.tool_calls = [mock_tool_call]

            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]
            mock_acompletion.return_value = mock_response

            mock_tool = Mock(spec=LLMTool)
            mock_tool.to_openai_tool.return_value = {
                "type": "function",
                "function": {"name": "get_weather"},
            }

            provider = LLMProvider(model="gpt-4o-mini", api_key="test-key")

            llm_response = await provider.get_llm_response(
                context_manager, tools=[mock_tool]
            )
            text_response = await provider.get_response(
                context_manager, tools=[mock_tool]
            )
            tool_calls = await provider.get_tool_calls(
                context_manager, tools=[mock_tool]
            )

            assert llm_response["text"] is None
            assert len(llm_response["tool_calls"]) == 1
            assert text_response is None
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "get_weather"
            assert tool_calls[0]["arguments"] == {"location": "Paris"}
