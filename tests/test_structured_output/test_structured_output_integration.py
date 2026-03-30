"""Tests for structured output integration in LLMBehaviour."""

import pytest
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel

from spade_llm.behaviour import LLMBehaviour
from spade_llm.structured_output import ReadyForStructuredOutputTool
from spade_llm.context import ContextManager
from tests.conftest import MockLLMProvider


class WeatherReport(BaseModel):
    """Sample structured output schema."""
    city: str
    temperature: float
    summary: str


class TripPlan(BaseModel):
    """Another sample schema."""
    destination: str
    days: int
    budget: float


class TestStructuredOutputBehaviourInit:
    """Test LLMBehaviour initialization with output_schema."""

    def test_init_with_output_schema(self):
        """Test that output_schema is stored."""
        provider = MockLLMProvider()
        behaviour = LLMBehaviour(llm_provider=provider, output_schema=WeatherReport)

        assert behaviour.output_schema is WeatherReport

    def test_init_without_output_schema(self):
        """Test default is None."""
        provider = MockLLMProvider()
        behaviour = LLMBehaviour(llm_provider=provider)

        assert behaviour.output_schema is None


class TestStructuredOutputWithoutTools:
    """Test structured output when no tools are registered (direct schema mode)."""

    @pytest.mark.asyncio
    async def test_structured_output_no_tools(self, mock_message):
        """Test structured output is returned as JSON when no tools are used."""
        report = WeatherReport(city="Valencia", temperature=25.0, summary="Sunny")
        provider = MockLLMProvider(structured_responses=[report])

        behaviour = LLMBehaviour(
            llm_provider=provider,
            output_schema=WeatherReport
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have sent the structured response as JSON
        behaviour.send.assert_called_once()
        sent = behaviour.send.call_args[0][0]
        assert "Valencia" in sent.body
        assert "25.0" in sent.body

        # Verify the provider was called with output_schema
        assert provider.call_history[0]["output_schema"] is WeatherReport


class TestStructuredOutputWithTools:
    """Test the two-phase pattern: tools + ready signal + structured generation."""

    @pytest.mark.asyncio
    async def test_ready_signal_injected_when_tools_and_schema(self, mock_message, mock_simple_tool):
        """Test that ReadyForStructuredOutputTool is injected when both tools and schema are provided."""
        report = WeatherReport(city="Madrid", temperature=30.0, summary="Hot")

        # Phase 1: LLM calls ready_for_structured_output tool
        # Phase 2: LLM generates structured output (no tools, with schema)
        provider = MockLLMProvider(
            tool_calls=[[{
                "id": "call_ready",
                "name": "ready_for_structured_output",
                "arguments": {}
            }]],
            structured_responses=[report]
        )

        behaviour = LLMBehaviour(
            llm_provider=provider,
            tools=[mock_simple_tool],
            output_schema=WeatherReport
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Provider should have been called exactly twice:
        # 1. With tools (including ready signal tool) => returns tool call
        # 2. Without tools, with output_schema => returns structured response
        assert provider.call_count == 2

        # First call should include the ready signal tool
        first_call_tools = provider.call_history[0]["tools"]
        assert "ready_for_structured_output" in first_call_tools

        # Second call should have no tools and output_schema set
        second_call = provider.call_history[1]
        assert second_call["tools"] == []
        assert second_call["output_schema"] is WeatherReport

        # Response should be sent as JSON
        behaviour.send.assert_called_once()
        sent = behaviour.send.call_args[0][0]
        assert "Madrid" in sent.body

    @pytest.mark.asyncio
    async def test_tool_use_then_ready_signal(self, mock_message, mock_simple_tool):
        """Test flow: use tool -> call ready signal -> get structured output."""
        report = WeatherReport(city="Barcelona", temperature=28.0, summary="Warm")

        provider = MockLLMProvider(
            tool_calls=[
                # First iteration: use a regular tool
                [{
                    "id": "call_tool1",
                    "name": "simple_tool",
                    "arguments": {"text": "get weather data"}
                }],
                # Second iteration: signal ready for structured output
                [{
                    "id": "call_ready",
                    "name": "ready_for_structured_output",
                    "arguments": {}
                }],
            ],
            structured_responses=[report]
        )

        behaviour = LLMBehaviour(
            llm_provider=provider,
            tools=[mock_simple_tool],
            output_schema=WeatherReport
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have called provider 3 times:
        # 1. With tools => returns simple_tool call
        # 2. With tools => returns ready signal call
        # 3. Without tools, with schema => returns structured output
        assert provider.call_count == 3

        # Final response should be structured
        behaviour.send.assert_called_once()
        sent = behaviour.send.call_args[0][0]
        assert "Barcelona" in sent.body

    @pytest.mark.asyncio
    async def test_ready_signal_with_other_tools_in_same_batch(self, mock_message, mock_simple_tool):
        """Test that when ready signal is in a batch with other tools, it is processed last."""
        report = WeatherReport(city="Lisbon", temperature=22.0, summary="Pleasant")

        provider = MockLLMProvider(
            tool_calls=[
                # Both a regular tool and the ready signal in the same batch
                [
                    {
                        "id": "call_ready",
                        "name": "ready_for_structured_output",
                        "arguments": {}
                    },
                    {
                        "id": "call_tool1",
                        "name": "simple_tool",
                        "arguments": {"text": "some data"}
                    },
                ],
            ],
            structured_responses=[report]
        )

        behaviour = LLMBehaviour(
            llm_provider=provider,
            tools=[mock_simple_tool],
            output_schema=WeatherReport
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # simple_tool should be in context results (executed before ready signal)
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        history = behaviour.context.get_conversation_history(conv_id)
        tool_results = [msg for msg in history if msg.get("role") == "tool"]
        # At least the simple_tool result and the ready signal result
        assert len(tool_results) >= 2

        # Verify order: simple_tool should be processed before ready_for_structured_output
        tool_names = [msg.get("tool_name") for msg in tool_results if msg.get("tool_name")]
        simple_idx = next(i for i, n in enumerate(tool_names) if n == "simple_tool")
        ready_idx = next(i for i, n in enumerate(tool_names) if n == "ready_for_structured_output")
        assert simple_idx < ready_idx, "simple_tool should be executed before ready_for_structured_output"

        # Response should be structured
        behaviour.send.assert_called_once()


class TestStructuredOutputFallbackAndErrors:
    """Test error handling and fallback behavior for structured output."""

    @pytest.mark.asyncio
    async def test_structured_output_fallback_to_text(self, mock_message, mock_simple_tool):
        """Test that when structured output fails after ready signal, text response is used."""
        # Provider returns ready signal tool call, then fails structured output (returns text instead)
        provider = MockLLMProvider(
            tool_calls=[[{
                "id": "call_ready",
                "name": "ready_for_structured_output",
                "arguments": {}
            }]],
            responses=["Fallback text response"],
            structured_responses=[]  # No structured responses available
        )

        behaviour = LLMBehaviour(
            llm_provider=provider,
            tools=[mock_simple_tool],
            output_schema=WeatherReport
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should still send a response (fallback to text)
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_output_provider_error(self, mock_message):
        """Test that provider errors propagate correctly with structured output."""
        provider = MockLLMProvider(should_error=True)

        behaviour = LLMBehaviour(
            llm_provider=provider,
            output_schema=WeatherReport
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send error message
        behaviour.send.assert_called_once()
        sent = behaviour.send.call_args[0][0]
        assert "Error" in sent.body or "error" in sent.body


class TestStructuredOutputContextManager:
    """Test output_schema methods on ContextManager."""

    def test_set_and_get_output_schema(self):
        """Test setting and getting output schema."""
        ctx = ContextManager()
        ctx._current_conversation_id = "conv1"

        ctx.set_output_schema(WeatherReport, "conv1")
        assert ctx.get_output_schema("conv1") is WeatherReport

    def test_get_output_schema_none_by_default(self):
        """Test that output schema is None by default."""
        ctx = ContextManager()
        assert ctx.get_output_schema("nonexistent") is None

    def test_set_output_schema_per_conversation(self):
        """Test that schemas are per-conversation."""
        ctx = ContextManager()
        ctx._current_conversation_id = "conv1"

        ctx.set_output_schema(WeatherReport, "conv1")
        ctx.set_output_schema(TripPlan, "conv2")

        assert ctx.get_output_schema("conv1") is WeatherReport
        assert ctx.get_output_schema("conv2") is TripPlan

    def test_set_output_schema_to_none(self):
        """Test clearing the output schema."""
        ctx = ContextManager()
        ctx._current_conversation_id = "conv1"

        ctx.set_output_schema(WeatherReport, "conv1")
        ctx.set_output_schema(None, "conv1")

        assert ctx.get_output_schema("conv1") is None
