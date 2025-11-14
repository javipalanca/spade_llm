"""Base provider for LLM integration."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..context import ContextManager
from ..tools import LLMTool

logger = logging.getLogger("spade_llm.providers")


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement,
    allowing for consistent interaction regardless of the underlying LLM service.

    Providers are responsible for communication with LLM services only.
    Tools and capabilities are managed at the agent/behaviour level.
    """

    def __init__(self):
        """Initialize the LLM provider."""
        pass

    @abstractmethod
    async def get_llm_response(
        self, context: ContextManager, tools: Optional[List[LLMTool]] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get complete response from the LLM including both text and tool calls.

        Args:
            context: The conversation context manager
            tools: Optional list of tools available for this specific call
            conversation_id: Optional conversation ID to retrieve specific conversation context

        Returns:
            Dictionary containing:
            - 'text': The text response (None if there are tool calls)
            - 'tool_calls': List of tool calls (empty if there are none)
        """
        pass

    # Legacy methods that delegate to the new unified method
    async def get_response(
        self, context: ContextManager, tools: Optional[List[LLMTool]] = None
    ) -> Optional[str]:
        """
        Get a response from the LLM based on the current context.

        Args:
            context: The conversation context manager
            tools: Optional list of tools available for this specific call

        Returns:
            The LLM's response as a string, or None if tool calls should be processed first
        """
        response = await self.get_llm_response(context, tools)
        return response.get("text")

    async def get_tool_calls(
        self, context: ContextManager, tools: Optional[List[LLMTool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tool calls from the LLM based on the current context.

        Args:
            context: The conversation context manager
            tools: Optional list of tools available for this specific call

        Returns:
            List of tool call specifications
        """
        response = await self.get_llm_response(context, tools)
        return response.get("tool_calls", [])
