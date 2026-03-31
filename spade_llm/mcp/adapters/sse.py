"""SSE adapter for MCP tools."""

import logging

from mcp.types import Tool

from ..config import SseServerConfig
from .base import MCPToolAdapter

logger = logging.getLogger(__name__)


class SseMCPToolAdapter(MCPToolAdapter):
    """Adapter for MCP tools using SSE transport.

    This adapter allows using MCP tools that communicate via HTTP with
    Server-Sent Events (SSE) with SPADE_LLM agents.
    """

    def __init__(self, server_config: SseServerConfig, tool: Tool):
        """Initialize the SSE MCP tool adapter.

        Args:
            server_config: Configuration for the MCP server.
            tool: The MCP tool to adapt.
        """
        if not isinstance(server_config, SseServerConfig):
            raise TypeError(f"Expected SseServerConfig, got {type(server_config)}")

        super().__init__(server_config=server_config, tool=tool)

        # Override the name with a more descriptive one
        self.name = f"{self.server_config.name}_{self.tool.name}"
