"""LLM tool implementation for SPADE agents."""

import asyncio
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger("spade_llm.tools")


class LLMTool:
    """
    A tool that can be invoked by an LLM.

    This class standardizes the definition, validation, and execution of tools
    that can be called by large language models during agent interactions.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
        strict: bool = False,
    ):
        """
        Initialize a new LLM tool.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            parameters: JSON Schema definition of the tool's parameters.
            func: The function to execute when this tool is invoked.
            strict: If True, enforce strict parameter validation.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.strict = strict

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary representation suitable for LLM APIs.

        Returns:
            A dictionary representation of the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI's tool format.

        Returns:
            A dictionary in OpenAI's tool format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            },
        }

    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the provided arguments.

        Args:
            **kwargs: Arguments to pass to the tool function.

        Returns:
            The result of the tool execution.
        """
        # Check if function is async
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            # Run synchronous function in a thread to avoid blocking
            return await asyncio.to_thread(self.func, **kwargs)
