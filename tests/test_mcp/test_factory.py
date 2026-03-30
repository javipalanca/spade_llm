"""Tests for MCP factory functions."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp.types import Tool

from spade_llm.mcp.config import SseServerConfig, StdioServerConfig
from spade_llm.mcp.factory import get_all_mcp_tools, get_mcp_server_tools, get_mcp_tool
from spade_llm.tools import LLMTool


class TestGetMCPServerTools:
    """Test get_mcp_server_tools function."""

    @pytest.mark.asyncio
    async def test_get_stdio_server_tools_success(self):
        """Test getting tools from stdio server successfully."""
        # Create mock tools
        mock_tool1 = Mock(spec=Tool)
        mock_tool1.name = "test_tool_1"
        mock_tool1.description = "First test tool"

        mock_tool2 = Mock(spec=Tool)
        mock_tool2.name = "test_tool_2"
        mock_tool2.description = "Second test tool"

        mock_tools = [mock_tool1, mock_tool2]

        # Create server config
        server_config = StdioServerConfig(name="test_server", command="python", args=["test_server.py"])

        # Mock MCPSession
        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=mock_tools)
            mock_session_class.return_value = mock_session

            # Mock StdioMCPToolAdapter
            with patch("spade_llm.mcp.factory.StdioMCPToolAdapter") as mock_adapter:
                mock_adapter.side_effect = lambda config, tool: Mock(spec=LLMTool, name=f"adapter_{tool.name}")

                result = await get_mcp_server_tools(server_config)

                # Should create session with config
                mock_session_class.assert_called_once_with(server_config)

                # Should get tools from session
                mock_session.get_tools.assert_called_once()

                # Should create adapters for each tool
                assert mock_adapter.call_count == 2
                mock_adapter.assert_any_call(server_config, mock_tool1)
                mock_adapter.assert_any_call(server_config, mock_tool2)

                # Should return list of adapted tools
                assert len(result) == 2
                assert all(isinstance(tool, Mock) for tool in result)

    @pytest.mark.asyncio
    async def test_get_sse_server_tools_success(self):
        """Test getting tools from SSE server successfully."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "sse_tool"
        mock_tool.description = "SSE test tool"

        # Create SSE server config
        server_config = SseServerConfig(name="sse_server", url="https://example.com/sse")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[mock_tool])
            mock_session_class.return_value = mock_session

            with patch("spade_llm.mcp.factory.SseMCPToolAdapter") as mock_adapter:
                mock_adapter.return_value = Mock(spec=LLMTool, name="sse_adapter")

                result = await get_mcp_server_tools(server_config)

                # Should use SSE adapter
                mock_adapter.assert_called_once_with(server_config, mock_tool)

                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_server_tools_session_error(self):
        """Test handling session errors."""
        server_config = StdioServerConfig(name="error_server", command="python")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(side_effect=Exception("Session error"))
            mock_session_class.return_value = mock_session

            with pytest.raises(RuntimeError, match="Failed to get tools from MCP server"):
                await get_mcp_server_tools(server_config)

    @pytest.mark.asyncio
    async def test_get_server_tools_empty_result(self):
        """Test getting empty tools list from server."""
        server_config = StdioServerConfig(name="empty_server", command="python")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[])
            mock_session_class.return_value = mock_session

            with patch("spade_llm.mcp.factory.StdioMCPToolAdapter"):
                result = await get_mcp_server_tools(server_config)

                assert result == []


class TestGetMCPTool:
    """Test get_mcp_tool function."""

    @pytest.mark.asyncio
    async def test_get_specific_tool_success(self):
        """Test getting a specific tool successfully."""
        # Create mock tools
        mock_tool1 = Mock(spec=Tool)
        mock_tool1.name = "tool_one"

        mock_tool2 = Mock(spec=Tool)
        mock_tool2.name = "target_tool"

        mock_tool3 = Mock(spec=Tool)
        mock_tool3.name = "tool_three"

        mock_tools = [mock_tool1, mock_tool2, mock_tool3]

        server_config = StdioServerConfig(name="test_server", command="python")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=mock_tools)
            mock_session_class.return_value = mock_session

            with patch("spade_llm.mcp.factory.StdioMCPToolAdapter") as mock_adapter:
                mock_adapter.return_value = Mock(spec=LLMTool, name="target_adapter")

                result = await get_mcp_tool(server_config, "target_tool")

                # Should find and adapt the correct tool
                mock_adapter.assert_called_once_with(server_config, mock_tool2)
                assert result is not None

    @pytest.mark.asyncio
    async def test_get_tool_not_found(self):
        """Test getting a tool that doesn't exist."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "existing_tool"

        server_config = StdioServerConfig(name="test_server", command="python")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[mock_tool])
            mock_session_class.return_value = mock_session

            result = await get_mcp_tool(server_config, "nonexistent_tool")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_tool_sse_server(self):
        """Test getting tool from SSE server."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "sse_tool"

        server_config = SseServerConfig(name="sse_server", url="https://example.com/sse")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[mock_tool])
            mock_session_class.return_value = mock_session

            with patch("spade_llm.mcp.factory.SseMCPToolAdapter") as mock_adapter:
                mock_adapter.return_value = Mock(spec=LLMTool, name="sse_adapter")

                result = await get_mcp_tool(server_config, "sse_tool")

                mock_adapter.assert_called_once_with(server_config, mock_tool)
                assert result is not None

    @pytest.mark.asyncio
    async def test_get_tool_session_error(self):
        """Test handling session error when getting specific tool."""
        server_config = StdioServerConfig(name="error_server", command="python")

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(side_effect=Exception("Connection failed"))
            mock_session_class.return_value = mock_session

            with pytest.raises(RuntimeError, match="Failed to get tool from MCP server"):
                await get_mcp_tool(server_config, "any_tool")


class TestGetAllMCPTools:
    """Test get_all_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_get_all_tools_success(self):
        """Test getting tools from multiple servers successfully."""
        # Create server configurations
        server1 = StdioServerConfig(name="server1", command="python")
        server2 = SseServerConfig(name="server2", url="https://example.com")
        server3 = StdioServerConfig(name="server3", command="node")

        server_configs = [server1, server2, server3]

        # Mock tools for each server
        tools1 = [Mock(spec=LLMTool, name="tool1"), Mock(spec=LLMTool, name="tool2")]
        tools2 = [Mock(spec=LLMTool, name="tool3")]
        tools3 = [Mock(spec=LLMTool, name="tool4"), Mock(spec=LLMTool, name="tool5"), Mock(spec=LLMTool, name="tool6")]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            mock_get_tools.side_effect = [tools1, tools2, tools3]

            result = await get_all_mcp_tools(server_configs)

            # Should call get_mcp_server_tools for each server
            assert mock_get_tools.call_count == 3
            mock_get_tools.assert_any_call(server1)
            mock_get_tools.assert_any_call(server2)
            mock_get_tools.assert_any_call(server3)

            # Should return all tools combined
            assert len(result) == 6
            assert all(tool in result for tool in tools1 + tools2 + tools3)

    @pytest.mark.asyncio
    async def test_get_all_tools_with_errors(self):
        """Test getting tools when some servers fail."""
        server1 = StdioServerConfig(name="server1", command="python")
        server2 = StdioServerConfig(name="server2", command="failing_command")
        server3 = StdioServerConfig(name="server3", command="node")

        server_configs = [server1, server2, server3]

        tools1 = [Mock(spec=LLMTool, name="tool1")]
        tools3 = [Mock(spec=LLMTool, name="tool3")]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            # Server 2 fails, others succeed
            mock_get_tools.side_effect = [tools1, RuntimeError("Server failed"), tools3]

            result = await get_all_mcp_tools(server_configs)

            # Should return tools from successful servers only
            assert len(result) == 2
            assert all(tool in result for tool in tools1 + tools3)

    @pytest.mark.asyncio
    async def test_get_all_tools_all_errors(self):
        """Test getting tools when all servers fail."""
        server1 = StdioServerConfig(name="server1", command="python")
        server2 = StdioServerConfig(name="server2", command="node")

        server_configs = [server1, server2]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            mock_get_tools.side_effect = [RuntimeError("First server failed"), RuntimeError("Second server failed")]

            result = await get_all_mcp_tools(server_configs)

            # Should return empty list
            assert result == []

    @pytest.mark.asyncio
    async def test_get_all_tools_empty_input(self):
        """Test getting tools with empty server list."""
        result = await get_all_mcp_tools([])

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_tools_mixed_server_types(self):
        """Test getting tools from mixed server types."""
        stdio_server = StdioServerConfig(name="stdio", command="python")
        sse_server = SseServerConfig(name="sse", url="https://example.com")

        server_configs = [stdio_server, sse_server]

        stdio_tools = [Mock(spec=LLMTool, name="stdio_tool")]
        sse_tools = [Mock(spec=LLMTool, name="sse_tool")]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            mock_get_tools.side_effect = [stdio_tools, sse_tools]

            result = await get_all_mcp_tools(server_configs)

            assert len(result) == 2
            assert stdio_tools[0] in result
            assert sse_tools[0] in result

    @pytest.mark.asyncio
    async def test_get_all_tools_some_empty_results(self):
        """Test getting tools when some servers return empty results."""
        server1 = StdioServerConfig(name="server1", command="python")
        server2 = StdioServerConfig(name="server2", command="node")
        server3 = StdioServerConfig(name="server3", command="go")

        server_configs = [server1, server2, server3]

        tools1 = [Mock(spec=LLMTool, name="tool1")]
        tools2 = []  # Empty result
        tools3 = [Mock(spec=LLMTool, name="tool3")]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            mock_get_tools.side_effect = [tools1, tools2, tools3]

            result = await get_all_mcp_tools(server_configs)

            # Should return non-empty results
            assert len(result) == 2
            assert tools1[0] in result
            assert tools3[0] in result

    @pytest.mark.asyncio
    async def test_get_all_tools_concurrent_execution(self):
        """Test that servers are queried concurrently."""
        import time

        server1 = StdioServerConfig(name="server1", command="python")
        server2 = StdioServerConfig(name="server2", command="node")

        server_configs = [server1, server2]

        async def slow_get_tools(config):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return [Mock(spec=LLMTool, name=f"tool_from_{config.name}")]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools", side_effect=slow_get_tools):
            start_time = time.time()
            result = await get_all_mcp_tools(server_configs)
            end_time = time.time()

            # Should take approximately 0.1 seconds (concurrent) rather than 0.2 (sequential)
            elapsed = end_time - start_time
            assert elapsed < 0.15  # Allow some margin for test execution

            assert len(result) == 2


class TestMCPFactoryEdgeCases:
    """Test edge cases for MCP factory functions."""

    @pytest.mark.asyncio
    async def test_get_server_tools_with_none_config(self):
        """Test getting tools with None config."""
        with pytest.raises(AttributeError):
            await get_mcp_server_tools(None)

    @pytest.mark.asyncio
    async def test_get_tool_with_none_config(self):
        """Test getting specific tool with None config."""
        with pytest.raises(AttributeError):
            await get_mcp_tool(None, "tool_name")

    @pytest.mark.asyncio
    async def test_get_tool_with_none_tool_name(self):
        """Test getting tool with None tool name."""
        server_config = StdioServerConfig(name="test", command="python")

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "actual_tool"

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[mock_tool])
            mock_session_class.return_value = mock_session

            result = await get_mcp_tool(server_config, None)

            # Should not find any tool
            assert result is None

    @pytest.mark.asyncio
    async def test_get_tool_with_empty_tool_name(self):
        """Test getting tool with empty tool name."""
        server_config = StdioServerConfig(name="test", command="python")

        mock_tool = Mock(spec=Tool)
        mock_tool.name = ""

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[mock_tool])
            mock_session_class.return_value = mock_session

            with patch("spade_llm.mcp.factory.StdioMCPToolAdapter") as mock_adapter:
                mock_adapter.return_value = Mock(spec=LLMTool)

                result = await get_mcp_tool(server_config, "")

                # Should find the tool with empty name
                assert result is not None

    @pytest.mark.asyncio
    async def test_large_number_of_servers(self):
        """Test with large number of servers."""
        servers = [StdioServerConfig(name=f"server_{i}", command="python") for i in range(50)]

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            mock_get_tools.side_effect = [[Mock(spec=LLMTool, name=f"tool_{i}")] for i in range(50)]

            result = await get_all_mcp_tools(servers)

            assert len(result) == 50
            assert mock_get_tools.call_count == 50

    @pytest.mark.asyncio
    async def test_duplicate_tool_names_across_servers(self):
        """Test handling duplicate tool names from different servers."""
        server1 = StdioServerConfig(name="server1", command="python")
        server2 = StdioServerConfig(name="server2", command="node")

        # Both servers have a tool with the same name
        tool1 = Mock(spec=LLMTool, name="common_tool")
        tool2 = Mock(spec=LLMTool, name="common_tool")

        with patch("spade_llm.mcp.factory.get_mcp_server_tools") as mock_get_tools:
            mock_get_tools.side_effect = [[tool1], [tool2]]

            result = await get_all_mcp_tools([server1, server2])

            # Should return both tools even with same name
            assert len(result) == 2
            assert tool1 in result
            assert tool2 in result

    @pytest.mark.asyncio
    async def test_tools_with_complex_metadata(self):
        """Test handling tools with complex metadata."""
        server_config = StdioServerConfig(name="test", command="python")

        # Create mock tool with complex structure
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "complex_tool"
        mock_tool.description = "A complex tool with metadata"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"param1": {"type": "string"}, "param2": {"type": "array", "items": {"type": "number"}}},
        }

        with patch("spade_llm.mcp.factory.MCPSession") as mock_session_class:
            mock_session = Mock()
            mock_session.get_tools = AsyncMock(return_value=[mock_tool])
            mock_session_class.return_value = mock_session

            with patch("spade_llm.mcp.factory.StdioMCPToolAdapter") as mock_adapter:
                mock_adapter.return_value = Mock(spec=LLMTool)

                result = await get_mcp_server_tools(server_config)

                # Should handle complex tool structure
                assert len(result) == 1
                mock_adapter.assert_called_once_with(server_config, mock_tool)
