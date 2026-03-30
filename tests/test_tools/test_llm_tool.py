"""Tests for LLMTool class."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from spade_llm.tools import LLMTool


class TestLLMToolInitialization:
    """Test LLMTool initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        def test_func():
            return "test result"
        
        tool = LLMTool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            func=test_func
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.func == test_func
    
    def test_init_with_complex_parameters(self):
        """Test initialization with complex parameter schema."""
        def complex_func(text: str, number: int, optional: bool = False):
            return f"{text}-{number}-{optional}"
        
        parameters = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input text"
                },
                "number": {
                    "type": "integer",
                    "description": "Input number",
                    "minimum": 0
                },
                "optional": {
                    "type": "boolean",
                    "description": "Optional flag",
                    "default": False
                }
            },
            "required": ["text", "number"]
        }
        
        tool = LLMTool(
            name="complex_tool",
            description="A complex test tool",
            parameters=parameters,
            func=complex_func
        )
        
        assert tool.parameters == parameters
        assert "required" in tool.parameters
        assert tool.parameters["required"] == ["text", "number"]


class TestLLMToolSerialization:
    """Test LLMTool serialization methods."""
    
    def test_to_dict(self, mock_simple_tool):
        """Test converting tool to dictionary."""
        tool_dict = mock_simple_tool.to_dict()
        
        expected = {
            "name": "simple_tool",
            "description": "A simple test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to process"
                    }
                },
                "required": ["text"]
            }
        }
        
        assert tool_dict == expected
    
    def test_to_openai_tool(self, mock_simple_tool):
        """Test converting tool to OpenAI format."""
        openai_format = mock_simple_tool.to_openai_tool()

        expected = {
            "type": "function",
            "function": {
                "name": "simple_tool",
                "description": "A simple test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to process"}
                    },
                    "required": ["text"],
                },
                "strict": False,
            },
        }
        
        assert openai_format == expected
    
    def test_to_openai_tool_empty_parameters(self):
        """Test OpenAI format with empty parameters."""
        def simple_func():
            return "result"
        
        tool = LLMTool(
            name="no_params_tool",
            description="Tool with no parameters",
            parameters={"type": "object", "properties": {}},
            func=simple_func
        )
        
        openai_format = tool.to_openai_tool()
        
        assert openai_format["function"]["parameters"]["properties"] == {}
        assert openai_format["type"] == "function"


class TestLLMToolExecution:
    """Test LLMTool execution functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_sync_function(self, mock_simple_tool):
        """Test executing a synchronous function."""
        result = await mock_simple_tool.execute(text="hello world")
        
        assert result == "Tool executed with: hello world"
    
    @pytest.mark.asyncio
    async def test_execute_sync_function_default_args(self, mock_simple_tool):
        """Test executing sync function with default arguments."""
        result = await mock_simple_tool.execute()
        
        assert result == "Tool executed with: default"
    
    @pytest.mark.asyncio
    async def test_execute_async_function(self, mock_async_tool):
        """Test executing an asynchronous function."""
        result = await mock_async_tool.execute(number=21)
        
        assert result == {"result": 42, "status": "success"}
    
    @pytest.mark.asyncio
    async def test_execute_async_function_default_args(self, mock_async_tool):
        """Test executing async function with default arguments."""
        result = await mock_async_tool.execute()
        
        assert result == {"result": 84, "status": "success"}  # 42 * 2
    
    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self):
        """Test executing tool with keyword arguments."""
        def multi_param_func(a: int, b: str, c: bool = True):
            return {"a": a, "b": b, "c": c}
        
        tool = LLMTool(
            name="multi_param",
            description="Multi-parameter tool",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "string"},
                    "c": {"type": "boolean"}
                }
            },
            func=multi_param_func
        )
        
        result = await tool.execute(a=123, b="test", c=False)
        
        assert result == {"a": 123, "b": "test", "c": False}
    
    @pytest.mark.asyncio
    async def test_execute_no_args(self):
        """Test executing tool with no arguments."""
        def no_args_func():
            return "no args result"
        
        tool = LLMTool(
            name="no_args",
            description="No arguments tool",
            parameters={"type": "object", "properties": {}},
            func=no_args_func
        )
        
        result = await tool.execute()
        
        assert result == "no args result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_error(self, mock_error_tool):
        """Test executing tool that raises an error."""
        with pytest.raises(ValueError, match="Intentional test error"):
            await mock_error_tool.execute()
    
    @pytest.mark.asyncio
    async def test_execute_async_tool_error(self):
        """Test executing async tool that raises an error."""
        async def async_error_func():
            raise RuntimeError("Async error")
        
        tool = LLMTool(
            name="async_error",
            description="Async error tool",
            parameters={"type": "object", "properties": {}},
            func=async_error_func
        )
        
        with pytest.raises(RuntimeError, match="Async error"):
            await tool.execute()
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_required_args(self):
        """Test executing tool with missing required arguments."""
        def required_args_func(required_param: str):
            return f"Got: {required_param}"
        
        tool = LLMTool(
            name="required_args",
            description="Tool with required args",
            parameters={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"}
                },
                "required": ["required_param"]
            },
            func=required_args_func
        )
        
        # This should raise TypeError due to missing required argument
        with pytest.raises(TypeError):
            await tool.execute()
    
    @pytest.mark.asyncio
    async def test_execute_with_extra_args(self):
        """Test executing tool with extra arguments."""
        def simple_func(param: str):
            return f"Got: {param}"
        
        tool = LLMTool(
            name="simple",
            description="Simple tool",
            parameters={"type": "object", "properties": {"param": {"type": "string"}}},
            func=simple_func
        )
        
        # Extra arguments should be ignored by the function
        with pytest.raises(TypeError):  # Unexpected keyword argument
            await tool.execute(param="value", extra="ignored")


class TestLLMToolIntegration:
    """Test LLMTool integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_tool_with_complex_return_types(self):
        """Test tool that returns complex data types."""
        def complex_return_func():
            return {
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "string": "text",
                "number": 42,
                "boolean": True,
                "null": None
            }
        
        tool = LLMTool(
            name="complex_return",
            description="Tool with complex return",
            parameters={"type": "object", "properties": {}},
            func=complex_return_func
        )
        
        result = await tool.execute()
        
        assert isinstance(result, dict)
        assert result["list"] == [1, 2, 3]
        assert result["dict"]["nested"] == "value"
        assert result["null"] is None
    
    @pytest.mark.asyncio
    async def test_tool_with_side_effects(self):
        """Test tool that has side effects."""
        call_log = []
        
        def side_effect_func(action: str):
            call_log.append(action)
            return f"Performed: {action}"
        
        tool = LLMTool(
            name="side_effect",
            description="Tool with side effects",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string"}
                }
            },
            func=side_effect_func
        )
        
        result1 = await tool.execute(action="first")
        result2 = await tool.execute(action="second")
        
        assert result1 == "Performed: first"
        assert result2 == "Performed: second"
        assert call_log == ["first", "second"]
    
    @pytest.mark.asyncio
    async def test_tool_execution_time(self):
        """Test that async execution doesn't block."""
        import time
        
        async def slow_async_func():
            await asyncio.sleep(0.01)  # Small delay
            return "slow result"
        
        def slow_sync_func():
            time.sleep(0.01)  # Small delay
            return "sync result"
        
        async_tool = LLMTool(
            name="slow_async",
            description="Slow async tool",
            parameters={"type": "object", "properties": {}},
            func=slow_async_func
        )
        
        sync_tool = LLMTool(
            name="slow_sync",
            description="Slow sync tool",
            parameters={"type": "object", "properties": {}},
            func=slow_sync_func
        )
        
        # Test that both complete successfully
        start_time = time.time()
        
        async_result = await async_tool.execute()
        sync_result = await sync_tool.execute()
        
        elapsed = time.time() - start_time
        
        assert async_result == "slow result"
        assert sync_result == "sync result"
        # Should complete in reasonable time (both delays are very small)
        assert elapsed < 1.0  # Should be much faster, but allowing margin


class TestLLMToolEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_tool_with_empty_name(self):
        """Test tool with empty name."""
        def test_func():
            return "result"
        
        tool = LLMTool(
            name="",
            description="Empty name tool",
            parameters={"type": "object", "properties": {}},
            func=test_func
        )
        
        assert tool.name == ""
        # Should still work for serialization
        tool_dict = tool.to_dict()
        assert tool_dict["name"] == ""
    
    def test_tool_with_none_description(self):
        """Test tool with None description."""
        def test_func():
            return "result"
        
        tool = LLMTool(
            name="test",
            description=None,
            parameters={"type": "object", "properties": {}},
            func=test_func
        )
        
        assert tool.description is None
        tool_dict = tool.to_dict()
        assert tool_dict["description"] is None
    
    def test_tool_with_none_parameters(self):
        """Test tool with None parameters."""
        def test_func():
            return "result"
        
        tool = LLMTool(
            name="test",
            description="Test tool",
            parameters=None,
            func=test_func
        )
        
        assert tool.parameters is None
        tool_dict = tool.to_dict()
        assert tool_dict["parameters"] is None
    
    @pytest.mark.asyncio
    async def test_tool_with_lambda_function(self):
        """Test tool with lambda function."""
        tool = LLMTool(
            name="lambda_tool",
            description="Tool using lambda",
            parameters={"type": "object", "properties": {}},
            func=lambda: "lambda result"
        )
        
        result = await tool.execute()
        assert result == "lambda result"
    
    @pytest.mark.asyncio
    async def test_tool_function_returns_none(self):
        """Test tool function that returns None."""
        def none_func():
            return None
        
        tool = LLMTool(
            name="none_tool",
            description="Tool that returns None",
            parameters={"type": "object", "properties": {}},
            func=none_func
        )
        
        result = await tool.execute()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_tool_function_no_return(self):
        """Test tool function with no explicit return."""
        def no_return_func():
            pass  # Implicitly returns None
        
        tool = LLMTool(
            name="no_return",
            description="Tool with no return",
            parameters={"type": "object", "properties": {}},
            func=no_return_func
        )
        
        result = await tool.execute()
        assert result is None
