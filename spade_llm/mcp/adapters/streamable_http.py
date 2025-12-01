"""Streamable HTTP adapter for MCP tools."""

import logging

from mcp.types import Tool

from ..config import StreamableHttpServerConfig
from .base import MCPToolAdapter

logger = logging.getLogger(__name__)


class StreamableHttpMCPToolAdapter(MCPToolAdapter):
    """Adapter for MCP tools using Streamable HTTP transport.

    This adapter allows using MCP tools that communicate via the
    Streamable HTTP protocol with SPADE_LLM agents. Streamable HTTP
    provides improved session management and stability compared to
    the traditional SSE transport.
    """

    def __init__(self, server_config: StreamableHttpServerConfig, tool: Tool):
        """Initialize the Streamable HTTP MCP tool adapter.

        Args:
            server_config: Configuration for the MCP server.
            tool: The MCP tool to adapt.
        """
        if not isinstance(server_config, StreamableHttpServerConfig):
            raise TypeError(
                f"Expected StreamableHttpServerConfig, got {type(server_config)}"
            )

        super().__init__(server_config=server_config, tool=tool)

        # Override the name with a more descriptive one
        self.name = f"{self.server_config.name}_{self.tool.name}"
