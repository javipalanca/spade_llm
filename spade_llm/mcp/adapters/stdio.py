"""STDIO adapter for MCP tools."""

import logging

from mcp.types import Tool

from ..config import StdioServerConfig
from .base import MCPToolAdapter

logger = logging.getLogger(__name__)


class StdioMCPToolAdapter(MCPToolAdapter):
    """Adapter for MCP tools using STDIO transport.

    This adapter allows using MCP tools that communicate via standard I/O
    with SPADE_LLM agents.
    """

    def __init__(self, server_config: StdioServerConfig, tool: Tool):
        """Initialize the STDIO MCP tool adapter.

        Args:
            server_config: Configuration for the MCP server.
            tool: The MCP tool to adapt.
        """
        if not isinstance(server_config, StdioServerConfig):
            raise TypeError(f"Expected StdioServerConfig, got {type(server_config)}")

        super().__init__(server_config=server_config, tool=tool)

        # Override the name with a more descriptive one
        self.name = f"{self.server_config.name}_{self.tool.name}"
