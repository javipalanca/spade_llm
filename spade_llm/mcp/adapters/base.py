"""Base adapter for MCP tools."""

import abc
import logging
from typing import Any, Dict

from mcp.types import CallToolResult, Tool

from ...tools.llm_tool import LLMTool
from ..config import MCPServerConfig
from ..session import MCPSession

logger = logging.getLogger(__name__)


class MCPToolAdapter(LLMTool, abc.ABC):
    """Base adapter for MCP tools.

    This abstract class provides the foundation for adapting MCP tools
    to the SPADE_LLM tool interface.
    """

    def __init__(
        self,
        server_config: MCPServerConfig,
        tool: Tool,
    ):
        """Initialize the MCP tool adapter.

        Args:
            server_config: Configuration for the MCP server.
            tool: The MCP tool to adapt.
        """
        self.server_config = server_config
        self.tool = tool
        self.session = MCPSession(server_config)

        # Process the tool schema to make it compatible with SPADE_LLM
        parameters = self._convert_schema(tool.inputSchema)

        # Initialize the LLMTool with the processed metadata
        super().__init__(
            name=f"{server_config.name}_{tool.name}",
            description=(tool.description or f"Tool '{tool.name}' from server '{server_config.name}'"),
            parameters=parameters,
            func=self._execute_tool,
        )

    def _convert_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an MCP JSON schema to SPADE_LLM format.

        Args:
            schema: The MCP tool's input schema.

        Returns:
            A JSON schema compatible with SPADE_LLM.
        """
        # MCP spec doesn't require the inputSchema to have `properties`,
        # but SPADE_LLM does
        result = schema.copy()
        if "properties" not in result:
            result["properties"] = {}

        # Ensure the schema type is an object
        if "type" not in result:
            result["type"] = "object"

        return result

    async def _execute_tool(self, **kwargs) -> Any:
        """Execute the MCP tool.

        Args:
            **kwargs: Arguments to pass to the tool.

        Returns:
            The result of the tool execution.

        Raises:
            RuntimeError: If tool execution fails.
        """
        try:
            logger.debug(f"Executing MCP tool {self.tool.name} with arguments: {kwargs}")

            # Call the tool via the MCP session
            result = await self.session.call_tool(self.tool.name, kwargs)

            # Process the result
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Error executing MCP tool {self.tool.name}: {e}")
            raise RuntimeError(f"Failed to execute MCP tool {self.tool.name}: {e}") from e

    def _process_result(self, result: CallToolResult) -> Any:
        """Process an MCP tool result into a suitable format for SPADE_LLM.

        Args:
            result: The raw MCP tool result.

        Returns:
            The processed result in a format suitable for SPADE_LLM.

        Raises:
            RuntimeError: If the result contains an error.
        """
        # Check for errors
        if result.isError:
            raise RuntimeError(f"MCP tool execution error: {result.content}")

        # Process the content
        if len(result.content) == 1:
            # Single content item, return it directly
            return result.content[0].model_dump()
        elif len(result.content) > 1:
            # Multiple content items, return as a list
            return [item.model_dump() for item in result.content]
        else:
            # No content, return None
            return None
