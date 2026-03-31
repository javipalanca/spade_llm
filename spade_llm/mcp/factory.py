"""Factory functions for creating MCP tools."""

import asyncio
import logging
from typing import List, Optional

from ..tools.llm_tool import LLMTool
from .adapters import (
    SseMCPToolAdapter,
    StdioMCPToolAdapter,
    StreamableHttpMCPToolAdapter,
)
from .config import (
    MCPServerConfig,
    SseServerConfig,
    StdioServerConfig,
    StreamableHttpServerConfig,
)
from .session import MCPSession

logger = logging.getLogger(__name__)


async def get_mcp_server_tools(server_config: MCPServerConfig) -> List[LLMTool]:
    """Get all tools from an MCP server as SPADE_LLM tools.

    This function connects to an MCP server, retrieves all available tools,
    and converts them to SPADE_LLM-compatible tool objects.

    Args:
        server_config: Configuration for the MCP server.

    Returns:
        A list of SPADE_LLM tools adapted from the MCP server tools.

    Raises:
        RuntimeError: If there is an error fetching the tools.
    """
    # Create a session to the server
    session = MCPSession(server_config)

    try:
        # Get all tools from the server
        tools = await session.get_tools()

        # Create appropriate adapters based on the server type
        if isinstance(server_config, StdioServerConfig):
            return [StdioMCPToolAdapter(server_config, tool) for tool in tools]
        elif isinstance(server_config, SseServerConfig):
            return [SseMCPToolAdapter(server_config, tool) for tool in tools]
        elif isinstance(server_config, StreamableHttpServerConfig):
            return [StreamableHttpMCPToolAdapter(server_config, tool) for tool in tools]
        else:
            raise ValueError(f"Unsupported server configuration type: {type(server_config)}")
    except Exception as e:
        logger.error(f"Error getting tools from MCP server {server_config.name}: {e}")
        raise RuntimeError(f"Failed to get tools from MCP server: {e}") from e


async def get_mcp_tool(server_config: MCPServerConfig, tool_name: str) -> Optional[LLMTool]:
    """Get a specific tool from an MCP server as a SPADE_LLM tool.

    This function connects to an MCP server, retrieves a specific tool,
    and converts it to a SPADE_LLM-compatible tool object.

    Args:
        server_config: Configuration for the MCP server.
        tool_name: The name of the tool to retrieve.

    Returns:
        A SPADE_LLM tool adapted from the MCP server tool, or None if not found.

    Raises:
        RuntimeError: If there is an error fetching the tool.
    """
    # Create a session to the server
    session = MCPSession(server_config)

    try:
        # Get all tools from the server
        tools = await session.get_tools()

        # Find the specified tool
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool is None:
            logger.warning(f"Tool '{tool_name}' not found on MCP server {server_config.name}")
            return None

        # Create appropriate adapter based on the server type
        if isinstance(server_config, StdioServerConfig):
            return StdioMCPToolAdapter(server_config, tool)
        elif isinstance(server_config, SseServerConfig):
            return SseMCPToolAdapter(server_config, tool)
        elif isinstance(server_config, StreamableHttpServerConfig):
            return StreamableHttpMCPToolAdapter(server_config, tool)
        else:
            raise ValueError(f"Unsupported server configuration type: {type(server_config)}")
    except Exception as e:
        logger.error(f"Error getting tool '{tool_name}' from MCP server {server_config.name}: {e}")
        raise RuntimeError(f"Failed to get tool from MCP server: {e}") from e


async def get_all_mcp_tools(server_configs: List[MCPServerConfig]) -> List[LLMTool]:
    """Get all tools from multiple MCP servers.

    This function connects to multiple MCP servers, retrieves all available tools,
    and converts them to SPADE_LLM-compatible tool objects.

    Args:
        server_configs: List of configurations for MCP servers.

    Returns:
        A list of SPADE_LLM tools adapted from all MCP server tools.
    """
    # Create tasks to get tools from each server
    tasks = [get_mcp_server_tools(config) for config in server_configs]

    # Run tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    all_tools = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error getting tools from server {server_configs[i].name}: {result}")
        else:
            all_tools.extend(result)

    return all_tools
