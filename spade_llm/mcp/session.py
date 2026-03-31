"""Session management for MCP server connections."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

import mcp
import mcp.types
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

# Try to import streamable_http, but don't fail if it's not available
try:
    from mcp.client.streamable_http import streamablehttp_client

    HAS_STREAMABLE_HTTP = True
except ImportError:
    HAS_STREAMABLE_HTTP = False

import logging

from .config import (
    MCPServerConfig,
    SseServerConfig,
    StdioServerConfig,
    StreamableHttpServerConfig,
)

logger = logging.getLogger(__name__)


def create_stdio_params(config: StdioServerConfig) -> StdioServerParameters:
    """Convert a StdioServerConfig to StdioServerParameters.

    Args:
        config: The configuration to convert.

    Returns:
        StdioServerParameters compatible with the MCP library.
    """
    return StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env,
        cwd=config.cwd,
        encoding=config.encoding,
        encoding_error_handler=config.encoding_error_handler,
    )


@asynccontextmanager
async def create_mcp_session(
    config: MCPServerConfig,
) -> AsyncGenerator[ClientSession, None]:
    """Create an MCP client session for a server.

    Args:
        config: The server configuration.

    Yields:
        A connected MCP client session.

    Raises:
        ValueError: If the configuration type is not supported.
        RuntimeError: If there is an error connecting to the server.
    """
    try:
        if isinstance(config, StdioServerConfig):
            stdio_params = create_stdio_params(config)
            async with stdio_client(stdio_params) as (read, write):
                async with ClientSession(
                    read_stream=read,
                    write_stream=write,
                    read_timeout_seconds=timedelta(seconds=config.read_timeout_seconds),
                ) as session:
                    yield session
        elif isinstance(config, SseServerConfig):
            async with sse_client(
                url=config.url,
                headers=config.headers,
                timeout=config.timeout,
                sse_read_timeout=config.sse_read_timeout,
            ) as (read, write):
                async with ClientSession(read_stream=read, write_stream=write) as session:
                    yield session
        elif isinstance(config, StreamableHttpServerConfig):
            if not HAS_STREAMABLE_HTTP:
                raise RuntimeError(
                    "Streamable HTTP transport is not available in the installed MCP version. "
                    "Please upgrade MCP to use this transport: pip install --upgrade mcp"
                )
            async with streamablehttp_client(
                url=config.url,
                headers=config.headers,
                timeout=timedelta(seconds=config.timeout),
                sse_read_timeout=timedelta(seconds=config.sse_read_timeout),
                terminate_on_close=config.terminate_on_close,
            ) as (read, write, session_id_callback):
                # TODO: Handle session_id_callback if needed in the future
                async with ClientSession(
                    read_stream=read,
                    write_stream=write,
                    read_timeout_seconds=timedelta(seconds=config.sse_read_timeout),
                ) as session:
                    yield session
        else:
            raise ValueError(f"Unsupported server configuration type: {type(config)}")
    except Exception as e:
        logger.error(f"Error connecting to MCP server {config.name}: {e}")
        raise RuntimeError(f"Failed to connect to MCP server: {e}") from e


class MCPSession:
    """Manages communication with MCP servers.

    This class provides a higher-level interface for working with MCP servers,
    including connection management and tools caching.
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize the MCP session.

        Args:
            config: The server configuration.
        """
        self.config = config
        self._tools_cache: Optional[List[mcp.types.Tool]] = None
        self._lock = asyncio.Lock()

    async def get_tools(self) -> List[mcp.types.Tool]:
        """Get the list of tools from the server.

        Returns:
            A list of available tools on the server.

        Raises:
            RuntimeError: If there is an error fetching the tools.
        """
        # Use cached tools if available and caching is enabled
        if self.config.cache_tools and self._tools_cache is not None:
            return self._tools_cache

        async with self._lock:
            # Check cache again in case another task filled it while waiting
            if self.config.cache_tools and self._tools_cache is not None:
                return self._tools_cache

            try:
                async with create_mcp_session(self.config) as session:
                    # Initialize the session
                    await session.initialize()

                    # Fetch the tools
                    tools_response = await session.list_tools()

                    # Cache the tools if requested
                    if self.config.cache_tools:
                        self._tools_cache = tools_response.tools

                    return tools_response.tools
            except Exception as e:
                logger.error(f"Error fetching tools from MCP server {self.config.name}: {e}")
                raise RuntimeError(f"Failed to fetch tools from MCP server: {e}") from e

    def invalidate_cache(self) -> None:
        """Invalidate the tools cache."""
        self._tools_cache = None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> mcp.types.CallToolResult:
        """Call a tool on the server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.

        Returns:
            The result of the tool call.

        Raises:
            RuntimeError: If there is an error calling the tool.
        """
        try:
            async with create_mcp_session(self.config) as session:
                # Initialize the session
                await session.initialize()

                # Call the tool
                return await session.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on MCP server {self.config.name}: {e}")
            raise RuntimeError(f"Failed to call tool {tool_name}: {e}") from e
