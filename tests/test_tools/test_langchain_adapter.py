"""Tests for LangChain adapter functionality."""

from unittest.mock import patch

import pytest

from spade_llm.tools.langchain_adapter import LangChainToolAdapter


class SimpleMockTool:
    """Simple mock tool for testing."""

    def __init__(self, name="test_tool", description="Test tool", args_schema=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.call_count = 0
        self.call_args = None

    def _call(self, **kwargs):
        self.call_count += 1
        self.call_args = kwargs
        return f"sync_result: {kwargs}"

    async def _acall(self, **kwargs):
        self.call_count += 1
        self.call_args = kwargs
        return f"async_result: {kwargs}"


class SimpleMockSchema:
    """Simple mock schema for testing."""

    def __init__(self, schema_dict):
        self._schema = schema_dict

    def schema(self):
        return self._schema


class TestLangChainToolAdapterBasic:
    """Test basic LangChain tool adapter functionality."""

    def test_init_basic_tool(self):
        """Test initialization with basic tool."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        assert adapter.name == "test_tool"
        assert adapter.description == "Test tool"
        assert adapter._expected_param == "input"
        assert adapter.lc_tool == tool

        # Test default schema
        expected_schema = {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Input for the test_tool tool"}},
            "required": ["input"],
        }
        assert adapter.parameters == expected_schema

    def test_init_with_schema(self):
        """Test initialization with schema."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(name="search_tool", args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        assert adapter.name == "search_tool"
        assert adapter.parameters == schema
        assert adapter._expected_param == "input"  # First property

    def test_init_with_single_property_schema(self):
        """Test initialization with single property schema."""
        schema = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        assert adapter._expected_param == "question"

    def test_init_with_multiple_properties_schema(self):
        """Test initialization with multiple properties schema."""
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}, "format": {"type": "string"}},
            "required": ["text"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        # Multiple properties should use "input"
        assert adapter._expected_param == "input"

    def test_init_with_unsupported_schema_type(self):
        """Test initialization with unsupported schema type."""
        mock_schema = type("MockSchema", (), {})()  # No schema method
        tool = SimpleMockTool(args_schema=mock_schema)

        with patch("spade_llm.tools.langchain_adapter.logger") as mock_logger:
            adapter = LangChainToolAdapter(tool)

            # Should log warning and use default schema
            mock_logger.warning.assert_called_once()
            assert "Unsupported schema type" in mock_logger.warning.call_args[0][0]

        # Should use default schema
        assert "input" in adapter.parameters["properties"]


class TestLangChainToolAdapterSchemaConversion:
    """Test schema conversion functionality."""

    def test_convert_schema_with_properties(self):
        """Test schema conversion with properties."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        result = adapter._convert_parameters_schema(tool)
        assert result == schema

    def test_convert_schema_without_properties(self):
        """Test schema conversion without properties."""
        schema = {"type": "string", "description": "A string"}
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        result = adapter._convert_parameters_schema(tool)
        expected = {"type": "object", "properties": schema, "required": []}
        assert result == expected

    def test_convert_schema_no_args_schema(self):
        """Test schema conversion with no args_schema."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        result = adapter._convert_parameters_schema(tool)
        assert result["type"] == "object"
        assert "input" in result["properties"]


class TestLangChainToolAdapterParameterTransformation:
    """Test parameter transformation functionality."""

    def test_transform_single_parameter(self):
        """Test transformation of single parameter."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        result = adapter._transform_parameters(query="test")
        assert result == {"input": "test"}

    def test_transform_common_parameters(self):
        """Test transformation of common parameter names."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        common_params = ["query", "question", "search", "text", "input", "q"]
        for param in common_params:
            result = adapter._transform_parameters(**{param: "test"})
            assert result == {"input": "test"}

    def test_transform_with_additional_params(self):
        """Test transformation with additional parameters."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        result = adapter._transform_parameters(query="test", max_results=5)
        assert result == {"input": "test", "max_results": 5}

    def test_transform_no_mapping(self):
        """Test transformation with no mapping needed."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        result = adapter._transform_parameters(custom_param="value")
        assert result == {"input": "value"}

    def test_transform_empty_params(self):
        """Test transformation with empty parameters."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        result = adapter._transform_parameters()
        assert result == {}

    def test_transform_schema_mismatch(self):
        """Test transformation with schema/kwargs mismatch."""
        schema = {
            "type": "object",
            "properties": {"param1": {"type": "string"}, "param2": {"type": "integer"}, "param3": {"type": "boolean"}},
            "required": ["param1"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        # Different length, should fall back to common param mapping
        result = adapter._transform_parameters(query="test", limit=5)
        assert result == {"input": "test", "limit": 5}


class TestLangChainToolAdapterIntegration:
    """Test integration functionality."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow."""
        tool = SimpleMockTool(name="calculator", description="Calculator tool")
        adapter = LangChainToolAdapter(tool)

        # Test initialization
        assert adapter.name == "calculator"
        assert adapter.description == "Calculator tool"

        # Test serialization methods
        tool_dict = adapter.to_dict()
        assert tool_dict["name"] == "calculator"
        assert tool_dict["description"] == "Calculator tool"

        openai_tool = adapter.to_openai_tool()
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "calculator"

        # Test execution
        result = await adapter.execute(input="2 + 2")
        assert "sync_result:" in result
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_workflow_with_schema(self):
        """Test workflow with complex schema."""
        schema = {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "operands": {"type": "array"},
                "precision": {"type": "integer"},
            },
            "required": ["operation", "operands"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(name="advanced_calc", args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        # Test schema conversion
        assert adapter.parameters == schema
        assert adapter._expected_param == "input"

        # Test execution
        result = await adapter.execute(operation="add", operands=[1, 2, 3], precision=2)
        assert "sync_result:" in result
        assert tool.call_count == 1

        # Verify parameters passed correctly
        assert tool.call_args["operation"] == "add"
        assert tool.call_args["operands"] == [1, 2, 3]
        assert tool.call_args["precision"] == 2

    @pytest.mark.asyncio
    async def test_parameter_edge_cases(self):
        """Test edge cases in parameter handling."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        # Test with None
        result = await adapter.execute(input=None)
        assert "sync_result:" in result
        assert tool.call_args == {"input": None}

        # Test with empty string
        result = await adapter.execute(input="")
        assert "sync_result:" in result
        assert tool.call_args == {"input": ""}

        # Test with complex data
        complex_data = {"nested": {"key": "value"}, "list": [1, 2, 3], "bool": True}
        result = await adapter.execute(input=complex_data)
        assert "sync_result:" in result
        assert tool.call_args == {"input": complex_data}


class TestLangChainToolAdapterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_get_expected_param_no_schema(self):
        """Test parameter name detection without schema."""
        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        param_name = adapter._get_expected_parameter_name()
        assert param_name == "input"

    def test_get_expected_param_no_schema_method(self):
        """Test parameter name detection with no schema method."""
        tool = SimpleMockTool()
        tool.args_schema = type("NoSchemaMethod", (), {})()
        adapter = LangChainToolAdapter(tool)

        param_name = adapter._get_expected_parameter_name()
        assert param_name == "input"

    def test_get_expected_param_empty_properties(self):
        """Test parameter name detection with empty properties."""
        schema = {"type": "object", "properties": {}, "required": []}
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        param_name = adapter._get_expected_parameter_name()
        assert param_name == "input"

    def test_get_expected_param_multiple_properties(self):
        """Test parameter name detection with multiple properties."""
        schema = {
            "type": "object",
            "properties": {"first": {"type": "string"}, "second": {"type": "integer"}},
            "required": ["first"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        param_name = adapter._get_expected_parameter_name()
        assert param_name == "input"  # Multiple properties default to "input"

    def test_inheritance_from_llm_tool(self):
        """Test that adapter inherits from LLMTool."""
        from spade_llm.tools.llm_tool import LLMTool

        tool = SimpleMockTool()
        adapter = LangChainToolAdapter(tool)

        assert isinstance(adapter, LLMTool)
        assert hasattr(adapter, "execute")
        assert hasattr(adapter, "to_dict")
        assert hasattr(adapter, "to_openai_tool")

    def test_transform_exact_match(self):
        """Test transform with exact parameter match."""
        schema = {
            "type": "object",
            "properties": {"search_query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["search_query"],
        }
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        result = adapter._transform_parameters(search_query="test", limit=10)
        assert result == {"search_query": "test", "limit": 10}

    def test_transform_no_properties(self):
        """Test transform with schema having no properties."""
        schema = {"type": "object", "required": []}
        mock_schema = SimpleMockSchema(schema)
        tool = SimpleMockTool(args_schema=mock_schema)
        adapter = LangChainToolAdapter(tool)

        result = adapter._transform_parameters(query="test")
        assert result == {"input": "test"}
