"""Tests for MCP session management."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager
from datetime import timedelta

import mcp.types
from mcp.client.stdio import StdioServerParameters
from mcp import ClientSession
from spade_llm.mcp.session import (
    create_stdio_params,
    create_mcp_session,
    MCPSession
)
from spade_llm.mcp.config import StdioServerConfig, SseServerConfig


class TestCreateStdioParams:
    """Test create_stdio_params function."""
    
    def test_create_stdio_params_minimal(self):
        """Test creating stdio params with minimal config."""
        config = StdioServerConfig(
            name="test_server",
            command="python"
        )
        
        params = create_stdio_params(config)
        
        assert isinstance(params, StdioServerParameters)
        assert params.command == "python"
        assert params.args == []
        assert params.env is None
        assert params.cwd is None
        assert params.encoding == "utf-8"
        assert params.encoding_error_handler == "strict"
    
    def test_create_stdio_params_full(self):
        """Test creating stdio params with full config."""
        config = StdioServerConfig(
            name="full_server",
            command="node",
            args=["server.js", "--port", "8080"],
            env={"NODE_ENV": "production", "PORT": "8080"},
            cwd="/app/server",
            encoding="utf-16",
            encoding_error_handler="ignore"
        )
        
        params = create_stdio_params(config)
        
        assert params.command == "node"
        assert params.args == ["server.js", "--port", "8080"]
        assert params.env == {"NODE_ENV": "production", "PORT": "8080"}
        assert params.cwd == "/app/server"
        assert params.encoding == "utf-16"
        assert params.encoding_error_handler == "ignore"
    
    def test_create_stdio_params_with_path_cwd(self):
        """Test creating stdio params with Path cwd."""
        from pathlib import Path
        
        config = StdioServerConfig(
            name="path_server",
            command="python",
            cwd=Path("/home/user/project")
        )
        
        params = create_stdio_params(config)
        assert params.cwd == Path("/home/user/project")
    
    def test_create_stdio_params_preserves_types(self):
        """Test that parameter types are preserved."""
        config = StdioServerConfig(
            name="type_test",
            command="python",
            args=["arg1", "arg2"],
            env={"KEY": "value"},
            read_timeout_seconds=10.5
        )
        
        params = create_stdio_params(config)
        
        assert isinstance(params.command, str)
        assert isinstance(params.args, list)
        assert isinstance(params.env, dict)
        assert isinstance(params.encoding, str)


class TestCreateMCPSession:
    """Test create_mcp_session context manager."""
    
    @pytest.mark.asyncio
    async def test_create_stdio_session_success(self):
        """Test creating stdio session successfully."""
        config = StdioServerConfig(
            name="test_server",
            command="python",
            args=["test.py"]
        )
        
        # Mock the stdio_client context manager
        mock_read_stream = Mock()
        mock_write_stream = Mock()
        
        @asynccontextmanager
        async def mock_stdio_client(params):
            yield mock_read_stream, mock_write_stream
        
        # Mock the ClientSession context manager
        mock_session = Mock(spec=ClientSession)
        
        @asynccontextmanager
        async def mock_client_session(read_stream, write_stream, read_timeout_seconds):
            yield mock_session
        
        with patch('spade_llm.mcp.session.stdio_client', mock_stdio_client):
            with patch('spade_llm.mcp.session.ClientSession', mock_client_session):
                async with create_mcp_session(config) as session:
                    assert session is mock_session
    
    @pytest.mark.asyncio
    async def test_create_sse_session_success(self):
        """Test creating SSE session successfully."""
        config = SseServerConfig(
            name="sse_server",
            url="https://example.com/sse",
            headers={"Authorization": "Bearer token"},
            timeout=30.0,
            sse_read_timeout=600.0
        )
        
        # Mock the sse_client context manager
        mock_read_stream = Mock()
        mock_write_stream = Mock()
        
        @asynccontextmanager
        async def mock_sse_client(url, headers, timeout, sse_read_timeout):
            # Verify parameters are passed correctly
            assert url == "https://example.com/sse"
            assert headers == {"Authorization": "Bearer token"}
            assert timeout == 30.0
            assert sse_read_timeout == 600.0
            yield mock_read_stream, mock_write_stream
        
        mock_session = Mock(spec=ClientSession)
        
        @asynccontextmanager
        async def mock_client_session(read_stream, write_stream):
            yield mock_session
        
        with patch('spade_llm.mcp.session.sse_client', mock_sse_client):
            with patch('spade_llm.mcp.session.ClientSession', mock_client_session):
                async with create_mcp_session(config) as session:
                    assert session is mock_session

    
    @pytest.mark.asyncio
    async def test_create_stdio_session_with_timeout(self):
        """Test creating stdio session with custom timeout."""
        config = StdioServerConfig(
            name="timeout_server",
            command="python",
            read_timeout_seconds=15.0
        )
        
        @asynccontextmanager
        async def mock_stdio_client(params):
            yield Mock(), Mock()
        
        @asynccontextmanager
        async def mock_client_session(read_stream, write_stream, read_timeout_seconds):
            # Verify timeout is converted correctly
            assert read_timeout_seconds == timedelta(seconds=15.0)
            yield Mock(spec=ClientSession)
        
        with patch('spade_llm.mcp.session.stdio_client', mock_stdio_client):
            with patch('spade_llm.mcp.session.ClientSession', mock_client_session):
                async with create_mcp_session(config) as session:
                    assert session is not None
    
    @pytest.mark.asyncio
    async def test_create_session_connection_error(self):
        """Test handling connection errors."""
        config = StdioServerConfig(
            name="error_server",
            command="nonexistent_command"
        )
        
        @asynccontextmanager
        async def mock_stdio_client_error(params):
            if False:
                yield  # Makes this a generator, but never reached
            raise ConnectionError("Failed to connect")
        
        with patch('spade_llm.mcp.session.stdio_client', mock_stdio_client_error):
            with pytest.raises(RuntimeError, match="Failed to connect to MCP server"):
                async with create_mcp_session(config):
                    pass
    
    @pytest.mark.asyncio
    async def test_create_session_cleanup_on_error(self):
        """Test that session is properly cleaned up on error."""
        config = StdioServerConfig(
            name="cleanup_test",
            command="python"
        )
        
        # Mock that ClientSession creation fails
        @asynccontextmanager
        async def mock_stdio_client(params):
            yield Mock(), Mock()
        
        @asynccontextmanager
        async def mock_client_session_error(read_stream, write_stream, read_timeout_seconds):
            if False:
                yield  # Makes this a generator, but never reached
            raise RuntimeError("Session creation failed")
        
        with patch('spade_llm.mcp.session.stdio_client', mock_stdio_client):
            with patch('spade_llm.mcp.session.ClientSession', mock_client_session_error):
                with pytest.raises(RuntimeError, match="Failed to connect to MCP server"):
                    async with create_mcp_session(config):
                        pass


class TestMCPSession:
    """Test MCPSession class."""
    
    def test_init(self):
        """Test MCPSession initialization."""
        config = StdioServerConfig(
            name="test_server",
            command="python",
            cache_tools=True
        )
        
        session = MCPSession(config)
        
        assert session.config is config
        assert session._tools_cache is None
        assert session._lock is not None
    
    @pytest.mark.asyncio
    async def test_get_tools_no_cache(self):
        """Test getting tools without caching."""
        config = StdioServerConfig(
            name="no_cache_server",
            command="python",
            cache_tools=False
        )
        
        # Create mock tools
        mock_tool1 = Mock(spec=mcp.types.Tool)
        mock_tool1.name = "tool1"
        
        mock_tool2 = Mock(spec=mcp.types.Tool)
        mock_tool2.name = "tool2"
        
        mock_tools = [mock_tool1, mock_tool2]
        
        # Mock tools response
        mock_tools_response = Mock()
        mock_tools_response.tools = mock_tools
        
        # Mock session and its methods
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            result = await session.get_tools()
            
            # Should initialize session and list tools
            mock_session.initialize.assert_called_once()
            mock_session.list_tools.assert_called_once()
            
            # Should return tools
            assert result == mock_tools
            
            # Should not cache tools
            assert session._tools_cache is None
    
    @pytest.mark.asyncio
    async def test_get_tools_with_cache_first_call(self):
        """Test getting tools with caching enabled (first call)."""
        config = StdioServerConfig(
            name="cache_server",
            command="python",
            cache_tools=True
        )
        
        mock_tools = [Mock(spec=mcp.types.Tool), Mock(spec=mcp.types.Tool)]
        mock_tools_response = Mock()
        mock_tools_response.tools = mock_tools
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            result = await session.get_tools()
            
            # Should fetch and cache tools
            assert result == mock_tools
            assert session._tools_cache == mock_tools
    
    @pytest.mark.asyncio
    async def test_get_tools_with_cache_subsequent_call(self):
        """Test getting tools with caching enabled (subsequent call)."""
        config = StdioServerConfig(
            name="cache_server",
            command="python",
            cache_tools=True
        )
        
        cached_tools = [Mock(spec=mcp.types.Tool), Mock(spec=mcp.types.Tool)]
        
        session = MCPSession(config)
        session._tools_cache = cached_tools  # Pre-populate cache
        
        # Should not create any session since cache is used
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            result = await session.get_tools()
            
            # Should return cached tools without creating session
            assert result == cached_tools
            mock_create_session.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_tools_error_handling(self):
        """Test error handling in get_tools."""
        config = StdioServerConfig(
            name="error_server",
            command="python"
        )
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context_error(config):
                if False:
                    yield  # Makes this a generator, but never reached
                raise ConnectionError("Connection failed")
            
            mock_create_session.return_value = mock_session_context_error(config)
            
            with pytest.raises(RuntimeError, match="Failed to fetch tools from MCP server"):
                await session.get_tools()
    
    @pytest.mark.asyncio
    async def test_get_tools_session_initialize_error(self):
        """Test error handling when session initialization fails."""
        config = StdioServerConfig(
            name="init_error_server",
            command="python"
        )
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock(side_effect=Exception("Init failed"))
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            with pytest.raises(RuntimeError, match="Failed to fetch tools from MCP server"):
                await session.get_tools()
    
    @pytest.mark.asyncio
    async def test_get_tools_list_tools_error(self):
        """Test error handling when list_tools fails."""
        config = StdioServerConfig(
            name="list_error_server",
            command="python"
        )
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=Exception("List tools failed"))
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            with pytest.raises(RuntimeError, match="Failed to fetch tools from MCP server"):
                await session.get_tools()
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        config = StdioServerConfig(
            name="cache_server",
            command="python",
            cache_tools=True
        )
        
        session = MCPSession(config)
        session._tools_cache = [Mock(), Mock()]  # Pre-populate cache
        
        assert session._tools_cache is not None
        
        session.invalidate_cache()
        
        assert session._tools_cache is None
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test calling a tool successfully."""
        config = StdioServerConfig(
            name="tool_server",
            command="python"
        )
        
        tool_name = "test_tool"
        tool_args = {"param1": "value1", "param2": 42}
        
        mock_result = Mock(spec=mcp.types.CallToolResult)
        mock_result.content = [{"type": "text", "text": "Tool executed successfully"}]
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            result = await session.call_tool(tool_name, tool_args)
            
            # Should initialize session and call tool
            mock_session.initialize.assert_called_once()
            mock_session.call_tool.assert_called_once_with(tool_name, tool_args)
            
            assert result is mock_result
    
    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        """Test error handling when calling tool."""
        config = StdioServerConfig(
            name="error_server",
            command="python"
        )
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context_error(config):
                if False:
                    yield  # Makes this a generator, but never reached
                raise Exception("Tool call failed")
            
            mock_create_session.return_value = mock_session_context_error(config)
            
            with pytest.raises(RuntimeError, match="Failed to call tool test_tool"):
                await session.call_tool("test_tool", {})
    
    @pytest.mark.asyncio
    async def test_call_tool_initialize_error(self):
        """Test error handling when tool call initialization fails."""
        config = StdioServerConfig(
            name="init_error_server",
            command="python"
        )
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock(side_effect=Exception("Init failed"))
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            with pytest.raises(RuntimeError, match="Failed to call tool test_tool"):
                await session.call_tool("test_tool", {})
    
    @pytest.mark.asyncio
    async def test_call_tool_execution_error(self):
        """Test error handling when tool execution fails."""
        config = StdioServerConfig(
            name="exec_error_server",
            command="python"
        )
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Tool execution failed"))
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            with pytest.raises(RuntimeError, match="Failed to call tool test_tool"):
                await session.call_tool("test_tool", {})


class TestMCPSessionConcurrency:
    """Test MCPSession concurrency handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_get_tools_with_cache(self):
        """Test concurrent calls to get_tools with caching."""
        config = StdioServerConfig(
            name="concurrent_server",
            command="python",
            cache_tools=True
        )
        
        mock_tools = [Mock(spec=mcp.types.Tool)]
        mock_tools_response = Mock()
        mock_tools_response.tools = mock_tools
        
        call_count = 0
        
        async def mock_list_tools():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate some delay
            return mock_tools_response
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=mock_list_tools)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            # Make multiple concurrent calls
            tasks = [session.get_tools() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
            # All should return the same tools
            for result in results:
                assert result == mock_tools
            
            # Should only call list_tools once due to locking
            assert call_count == 1
            
            # Cache should be populated
            assert session._tools_cache == mock_tools
    


class TestMCPSessionEdgeCases:
    """Test edge cases for MCP session management."""
    
    @pytest.mark.asyncio
    async def test_get_tools_empty_response(self):
        """Test getting tools with empty response."""
        config = StdioServerConfig(name="empty", command="python")
        
        mock_tools_response = Mock()
        mock_tools_response.tools = []
        
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            result = await session.get_tools()
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_call_tool_with_empty_args(self):
        """Test calling tool with empty arguments."""
        config = StdioServerConfig(name="test", command="python")
        
        mock_result = Mock(spec=mcp.types.CallToolResult)
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            result = await session.call_tool("empty_args_tool", {})
            
            mock_session.call_tool.assert_called_once_with("empty_args_tool", {})
            assert result is mock_result
    
    @pytest.mark.asyncio
    async def test_call_tool_with_complex_args(self):
        """Test calling tool with complex arguments."""
        config = StdioServerConfig(name="test", command="python")
        
        complex_args = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"}
            },
            "array": ["a", "b", "c"],
            "number": 42.5,
            "boolean": True,
            "null": None
        }
        
        mock_result = Mock(spec=mcp.types.CallToolResult)
        mock_session = Mock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        session = MCPSession(config)
        
        with patch('spade_llm.mcp.session.create_mcp_session') as mock_create_session:
            @asynccontextmanager
            async def mock_session_context(config):
                yield mock_session
            
            mock_create_session.return_value = mock_session_context(config)
            
            result = await session.call_tool("complex_tool", complex_args)
            
            mock_session.call_tool.assert_called_once_with("complex_tool", complex_args)
            assert result is mock_result
    
    def test_session_with_different_config_types(self):
        """Test MCPSession with different configuration types."""
        stdio_config = StdioServerConfig(name="stdio", command="python") 
        sse_config = SseServerConfig(name="sse", url="https://example.com")
        
        stdio_session = MCPSession(stdio_config)
        sse_session = MCPSession(sse_config)
        
        assert stdio_session.config is stdio_config
        assert sse_session.config is sse_config
        
        # Both should have separate locks and caches
        assert stdio_session._lock is not sse_session._lock
        assert stdio_session._tools_cache is None
        assert sse_session._tools_cache is None
