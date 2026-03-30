"""Tests for the ReadyForStructuredOutputTool."""

import pytest

from pydantic import BaseModel

from spade_llm.structured_output import ReadyForStructuredOutputTool


class SampleSchema(BaseModel):
    """Sample schema for testing."""
    name: str
    age: int
    active: bool = True


class TestReadyForStructuredOutputToolInit:
    """Test ReadyForStructuredOutputTool initialization."""

    def test_init_basic(self):
        """Test basic initialization with a schema."""
        tool = ReadyForStructuredOutputTool(SampleSchema)

        assert tool.name == ReadyForStructuredOutputTool.TOOL_NAME
        assert tool.name == "ready_for_structured_output"
        assert tool.output_schema is SampleSchema
        assert tool.func is not None

    def test_init_description_contains_signal_info(self):
        """Test that description explains the signaling purpose."""
        tool = ReadyForStructuredOutputTool(SampleSchema)

        assert "signal" in tool.description.lower() or "gathered enough context" in tool.description.lower()

    def test_init_no_parameters_required(self):
        """Test that the tool requires no parameters (it's a signal)."""
        tool = ReadyForStructuredOutputTool(SampleSchema)

        assert tool.parameters["properties"] == {}
        assert tool.parameters["required"] == []

    def test_init_strict_is_false(self):
        """Test that strict mode is disabled."""
        tool = ReadyForStructuredOutputTool(SampleSchema)

        assert tool.strict is False

    def test_tool_name_constant(self):
        """Test that TOOL_NAME is consistent."""
        assert ReadyForStructuredOutputTool.TOOL_NAME == "ready_for_structured_output"


class TestReadyForStructuredOutputToolSerialization:
    """Test serialization methods."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        tool = ReadyForStructuredOutputTool(SampleSchema)
        tool_dict = tool.to_dict()

        assert tool_dict["name"] == "ready_for_structured_output"
        assert "description" in tool_dict
        assert tool_dict["parameters"]["properties"] == {}

    def test_to_openai_tool(self):
        """Test converting to OpenAI tool format."""
        tool = ReadyForStructuredOutputTool(SampleSchema)
        openai_format = tool.to_openai_tool()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "ready_for_structured_output"
        assert openai_format["function"]["parameters"]["properties"] == {}
        assert openai_format["function"]["strict"] is False


class TestReadyForStructuredOutputToolExecution:
    """Test tool execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_ready_status(self):
        """Test that executing the tool returns a ready status."""
        tool = ReadyForStructuredOutputTool(SampleSchema)
        result = await tool.execute()

        assert result["status"] == "ready"
        assert "SampleSchema" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_with_kwargs_still_works(self):
        """Test that extra kwargs don't break execution."""
        tool = ReadyForStructuredOutputTool(SampleSchema)
        result = await tool.execute(extra_param="ignored")

        assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_execute_with_different_schema(self):
        """Test execution with a different schema."""
        class TripPlan(BaseModel):
            destination: str
            days: int

        tool = ReadyForStructuredOutputTool(TripPlan)
        result = await tool.execute()

        assert result["status"] == "ready"
        assert "TripPlan" in result["message"]
