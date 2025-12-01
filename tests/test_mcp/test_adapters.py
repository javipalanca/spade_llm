"""Tests for MCP tool adapters."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from mcp.types import Tool, CallToolResult

from spade_llm.mcp.adapters import MCPToolAdapter, StdioMCPToolAdapter, SseMCPToolAdapter
from spade_llm.mcp.config import StdioServerConfig, SseServerConfig
from spade_llm.tools import LLMTool


class TestMCPToolAdapterBase:
    """Test MCPToolAdapter base class."""
    
    def test_is_abstract(self):
        """Test that MCPToolAdapter is abstract."""
        # Should not be able to instantiate directly since it inherits from abc.ABC
        # But we can create a concrete subclass for testing
        
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        # Create mock tool and config
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Should inherit from LLMTool
            assert isinstance(adapter, LLMTool)
            assert adapter.name == "test_server_test_tool"
            assert adapter.description == "Test tool description"
    
    def test_init_with_description(self):
        """Test initialization with tool description."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "described_tool"
        mock_tool.description = "A well-described tool"
        mock_tool.inputSchema = {"type": "object", "properties": {"param": {"type": "string"}}}
        
        mock_config = Mock()
        mock_config.name = "described_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            assert adapter.name == "described_server_described_tool"
            assert adapter.description == "A well-described tool"
    
    def test_init_without_description(self):
        """Test initialization without tool description."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "undescribed_tool"
        mock_tool.description = None
        mock_tool.inputSchema = {"type": "object"}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            assert adapter.description == "Tool 'undescribed_tool' from server 'test_server'"
    
    def test_convert_schema_complete(self):
        """Test schema conversion with complete schema."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        input_schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "number", "minimum": 0}
            },
            "required": ["param1"]
        }
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "schema_tool"
        mock_tool.description = "Tool with schema"
        mock_tool.inputSchema = input_schema
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Schema should be preserved
            assert adapter.parameters == input_schema
    
    def test_convert_schema_missing_properties(self):
        """Test schema conversion when properties are missing."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        input_schema = {
            "type": "object",
            "required": ["param1"]
        }
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "schema_tool"
        mock_tool.description = "Tool with incomplete schema"
        mock_tool.inputSchema = input_schema
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Should add empty properties
            expected_schema = {
                "type": "object",
                "required": ["param1"],
                "properties": {}
            }
            assert adapter.parameters == expected_schema
    
    def test_convert_schema_missing_type(self):
        """Test schema conversion when type is missing."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        input_schema = {
            "properties": {
                "param": {"type": "string"}
            }
        }
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "schema_tool"
        mock_tool.description = "Tool with no type"
        mock_tool.inputSchema = input_schema
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Should add object type
            expected_schema = {
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
            assert adapter.parameters == expected_schema
    
    def test_convert_schema_empty(self):
        """Test schema conversion with empty schema."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        input_schema = {}
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "empty_schema_tool"
        mock_tool.description = "Tool with empty schema"
        mock_tool.inputSchema = input_schema
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Should add both type and properties
            expected_schema = {
                "type": "object",
                "properties": {}
            }
            assert adapter.parameters == expected_schema


class TestMCPToolAdapterExecution:
    """Test MCP tool adapter execution functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        # Create mock tool result
        mock_content = Mock()
        mock_content.model_dump.return_value = {"type": "text", "text": "Success result"}
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content]
        
        # Create mock session
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        # Create adapter
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Execute the tool
            result = await adapter._execute_tool(param1="value1", param2=42)
            
            # Should call session.call_tool with correct parameters
            mock_session.call_tool.assert_called_once_with("test_tool", {"param1": "value1", "param2": 42})
            
            # Should return processed result
            assert result == {"type": "text", "text": "Success result"}
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_error_result(self):
        """Test tool execution with error result."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = True
        mock_result.content = "Tool execution failed"
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "error_tool"
        mock_tool.description = "Tool that errors"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            with pytest.raises(RuntimeError, match="MCP tool execution error"):
                await adapter._execute_tool()
    
    @pytest.mark.asyncio
    async def test_execute_tool_session_error(self):
        """Test tool execution with session error."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Session error"))
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "session_error_tool"
        mock_tool.description = "Tool with session error"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            with pytest.raises(RuntimeError, match="Failed to execute MCP tool session_error_tool"):
                await adapter._execute_tool()
    
    def test_process_result_single_content(self):
        """Test processing result with single content item."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_content = Mock()
        mock_content.model_dump.return_value = {"type": "text", "text": "Single result"}
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content]
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "single_tool"
        mock_tool.description = "Single result tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            result = adapter._process_result(mock_result)
            
            assert result == {"type": "text", "text": "Single result"}
    
    def test_process_result_multiple_content(self):
        """Test processing result with multiple content items."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_content1 = Mock()
        mock_content1.model_dump.return_value = {"type": "text", "text": "First result"}
        
        mock_content2 = Mock()
        mock_content2.model_dump.return_value = {"type": "text", "text": "Second result"}
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content1, mock_content2]
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "multi_tool"
        mock_tool.description = "Multiple result tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            result = adapter._process_result(mock_result)
            
            expected = [
                {"type": "text", "text": "First result"},
                {"type": "text", "text": "Second result"}
            ]
            assert result == expected
    
    def test_process_result_no_content(self):
        """Test processing result with no content."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = []
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "empty_tool"
        mock_tool.description = "Empty result tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            result = adapter._process_result(mock_result)
            
            assert result is None
    
    def test_process_result_with_error(self):
        """Test processing result with error flag."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = True
        mock_result.content = "Error message"
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "error_tool"
        mock_tool.description = "Error tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            with pytest.raises(RuntimeError, match="MCP tool execution error"):
                adapter._process_result(mock_result)


class TestStdioMCPToolAdapter:
    """Test StdioMCPToolAdapter class."""
    
    def test_init_success(self):
        """Test successful initialization of stdio adapter."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "stdio_tool"
        mock_tool.description = "STDIO test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        config = StdioServerConfig(
            name="stdio_server",
            command="python",
            args=["server.py"]
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = StdioMCPToolAdapter(config, mock_tool)
            
            assert isinstance(adapter, MCPToolAdapter)
            assert adapter.name == "stdio_server_stdio_tool"
            assert adapter.server_config is config
            assert adapter.tool is mock_tool
    
    def test_init_wrong_config_type(self):
        """Test initialization with wrong config type."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        # Use SSE config instead of STDIO config
        wrong_config = SseServerConfig(
            name="sse_server",
            url="https://example.com"
        )
        
        with pytest.raises(TypeError, match="Expected StdioServerConfig"):
            StdioMCPToolAdapter(wrong_config, mock_tool)
    
    def test_init_with_complex_config(self):
        """Test initialization with complex stdio config."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "complex_tool"
        mock_tool.description = "Complex STDIO tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "options": {"type": "object"}
            },
            "required": ["input"]
        }
        
        config = StdioServerConfig(
            name="complex_server",
            command="node",
            args=["complex_server.js", "--verbose"],
            env={"NODE_ENV": "development"},
            cwd="/opt/servers",
            encoding="utf-8",
            read_timeout_seconds=30.0,
            cache_tools=True
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = StdioMCPToolAdapter(config, mock_tool)
            
            assert adapter.name == "complex_server_complex_tool"
            assert adapter.description == "Complex STDIO tool"
            
            # Should preserve complex schema
            expected_schema = {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "options": {"type": "object"}
                },
                "required": ["input"]
            }
            assert adapter.parameters == expected_schema
    
    @pytest.mark.asyncio
    async def test_execute_tool_integration(self):
        """Test tool execution through stdio adapter."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "integration_tool"
        mock_tool.description = "Integration test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {"param": {"type": "string"}}}
        
        config = StdioServerConfig(
            name="integration_server",
            command="python"
        )
        
        # Mock the execution result
        mock_content = Mock()
        mock_content.model_dump.return_value = {
            "type": "text",
            "text": "STDIO execution successful"
        }
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content]
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = StdioMCPToolAdapter(config, mock_tool)
            
            # Execute through the adapter
            result = await adapter.execute(param="test_value")
            
            # Should call the underlying tool with correct parameters
            mock_session.call_tool.assert_called_once_with("integration_tool", {"param": "test_value"})
            
            # Should return the processed result
            assert result == {"type": "text", "text": "STDIO execution successful"}


class TestSseMCPToolAdapter:
    """Test SseMCPToolAdapter class."""
    
    def test_init_success(self):
        """Test successful initialization of SSE adapter."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "sse_tool"
        mock_tool.description = "SSE test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        config = SseServerConfig(
            name="sse_server",
            url="https://api.example.com/sse"
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = SseMCPToolAdapter(config, mock_tool)
            
            assert isinstance(adapter, MCPToolAdapter)
            assert adapter.name == "sse_server_sse_tool"
            assert adapter.server_config is config
            assert adapter.tool is mock_tool
    
    def test_init_wrong_config_type(self):
        """Test initialization with wrong config type."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        # Use STDIO config instead of SSE config
        wrong_config = StdioServerConfig(
            name="stdio_server",
            command="python"
        )
        
        with pytest.raises(TypeError, match="Expected SseServerConfig"):
            SseMCPToolAdapter(wrong_config, mock_tool)
    
    def test_init_with_complex_config(self):
        """Test initialization with complex SSE config."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "complex_sse_tool"
        mock_tool.description = "Complex SSE tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "array", "items": {"type": "string"}}
            }
        }
        
        config = SseServerConfig(
            name="complex_sse_server",
            url="https://api.complex.com/sse/events",
            headers={
                "Authorization": "Bearer token123",
                "Accept": "text/event-stream"
            },
            timeout=60.0,
            sse_read_timeout=900.0,
            cache_tools=True
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = SseMCPToolAdapter(config, mock_tool)
            
            assert adapter.name == "complex_sse_server_complex_sse_tool"
            assert adapter.description == "Complex SSE tool"
            
            # Should preserve complex schema
            expected_schema = {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "array", "items": {"type": "string"}}
                }
            }
            assert adapter.parameters == expected_schema
    
    @pytest.mark.asyncio
    async def test_execute_tool_integration(self):
        """Test tool execution through SSE adapter."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "sse_integration_tool"
        mock_tool.description = "SSE integration test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            }
        }
        
        config = SseServerConfig(
            name="sse_integration_server",
            url="https://sse.example.com/events"
        )
        
        # Mock the execution result
        mock_content = Mock()
        mock_content.model_dump.return_value = {
            "type": "text",
            "text": "SSE execution successful"
        }
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content]
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = SseMCPToolAdapter(config, mock_tool)
            
            # Execute through the adapter
            result = await adapter.execute(message="test message")
            
            # Should call the underlying tool with correct parameters
            mock_session.call_tool.assert_called_once_with(
                "sse_integration_tool", 
                {"message": "test message"}
            )
            
            # Should return the processed result
            assert result == {"type": "text", "text": "SSE execution successful"}


class TestMCPAdaptersEdgeCases:
    """Test edge cases for MCP adapters."""
    
    def test_adapter_with_very_long_names(self):
        """Test adapter with very long server and tool names."""
        long_server_name = "a" * 100
        long_tool_name = "b" * 100
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = long_tool_name
        mock_tool.description = "Long name tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        config = StdioServerConfig(
            name=long_server_name,
            command="python"
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = StdioMCPToolAdapter(config, mock_tool)
            
            expected_name = f"{long_server_name}_{long_tool_name}"
            assert adapter.name == expected_name
            assert len(adapter.name) == len(expected_name)
    
    def test_adapter_with_special_characters_in_names(self):
        """Test adapter with special characters in names."""
        special_server_name = "server-with_special.chars@123"
        special_tool_name = "tool#with$special%chars"
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = special_tool_name
        mock_tool.description = "Special chars tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        config = SseServerConfig(
            name=special_server_name,
            url="https://example.com"
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = SseMCPToolAdapter(config, mock_tool)
            
            expected_name = f"{special_server_name}_{special_tool_name}"
            assert adapter.name == expected_name
    
    def test_adapter_with_empty_description(self):
        """Test adapter with empty string description."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "empty_desc_tool"
        mock_tool.description = ""
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        config = StdioServerConfig(
            name="test_server",
            command="python"
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = StdioMCPToolAdapter(config, mock_tool)
            
            # Empty description should use default format
            assert adapter.description == "Tool 'empty_desc_tool' from server 'test_server'"
    
    def test_adapter_with_complex_nested_schema(self):
        """Test adapter with deeply nested schema."""
        complex_schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "deep_param": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["level1"]
        }
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "nested_tool"
        mock_tool.description = "Deeply nested tool"
        mock_tool.inputSchema = complex_schema
        
        config = StdioServerConfig(
            name="nested_server",
            command="python"
        )
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            adapter = StdioMCPToolAdapter(config, mock_tool)
            
            # Should preserve the complex nested structure
            assert adapter.parameters == complex_schema
    
    @pytest.mark.asyncio
    async def test_adapter_execute_with_none_parameters(self):
        """Test adapter execution with None parameters."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "none_param_tool"
        mock_tool.description = "Tool with None params"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        mock_content = Mock()
        mock_content.model_dump.return_value = {"result": "success"}
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content]
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Execute with no parameters
            result = await adapter._execute_tool()
            
            # Should call tool with empty dict
            mock_session.call_tool.assert_called_once_with("none_param_tool", {})
            assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_adapter_execute_with_mixed_parameter_types(self):
        """Test adapter execution with mixed parameter types."""
        class ConcreteMCPAdapter(MCPToolAdapter):
            pass
        
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "mixed_param_tool"
        mock_tool.description = "Tool with mixed params"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_config = Mock()
        mock_config.name = "test_server"
        
        mock_content = Mock()
        mock_content.model_dump.return_value = {"result": "mixed success"}
        
        mock_result = Mock(spec=CallToolResult)
        mock_result.isError = False
        mock_result.content = [mock_content]
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('spade_llm.mcp.adapters.base.MCPSession', return_value=mock_session):
            adapter = ConcreteMCPAdapter(mock_config, mock_tool)
            
            # Execute with mixed parameter types
            mixed_params = {
                "string_param": "hello",
                "number_param": 42,
                "boolean_param": True,
                "null_param": None,
                "array_param": [1, 2, 3],
                "object_param": {"nested": "value"}
            }
            
            result = await adapter._execute_tool(**mixed_params)
            
            # Should call tool with all parameters
            mock_session.call_tool.assert_called_once_with("mixed_param_tool", mixed_params)
            assert result == {"result": "mixed success"}
    
    def test_adapter_inheritance_chain(self):
        """Test that adapters maintain proper inheritance chain."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "inheritance_tool"
        mock_tool.description = "Inheritance test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        stdio_config = StdioServerConfig(name="stdio", command="python")
        sse_config = SseServerConfig(name="sse", url="https://example.com")
        
        with patch('spade_llm.mcp.adapters.base.MCPSession'):
            stdio_adapter = StdioMCPToolAdapter(stdio_config, mock_tool)
            sse_adapter = SseMCPToolAdapter(sse_config, mock_tool)
            
            # Both should inherit from MCPToolAdapter and LLMTool
            assert isinstance(stdio_adapter, MCPToolAdapter)
            assert isinstance(stdio_adapter, LLMTool)
            assert isinstance(sse_adapter, MCPToolAdapter)
            assert isinstance(sse_adapter, LLMTool)
            
            # Should have proper method resolution order
            assert hasattr(stdio_adapter, '_execute_tool')
            assert hasattr(stdio_adapter, 'execute')
            assert hasattr(sse_adapter, '_execute_tool')
            assert hasattr(sse_adapter, 'execute')
