"""Tool that signals when LLM has enough context to generate structured output."""

import logging
from typing import Any, Dict, Type

from pydantic import BaseModel

from ..tools import LLMTool

logger = logging.getLogger("spade_llm.structured_output")


class ReadyForStructuredOutputTool(LLMTool):
    """
    A signal tool that indicates the LLM has gathered enough context.

    When this tool is called, the system switches to the structured output
    parsing API for constrained generation of the required schema.
    """

    TOOL_NAME = "ready_for_structured_output"

    def __init__(self, output_schema: Type[BaseModel]):
        """
        Initialize the ready_for_structured_output signal tool.

        Args:
            output_schema: Pydantic model defining the expected response structure
        """
        self.output_schema = output_schema
        schema_name = output_schema.__name__

        # Create a simple signaling function
        async def ready_for_structured_output() -> str:
            """Signal that you have gathered enough context to provide a structured response.

            Call this tool when you have all the information needed to generate
            the final structured output. You MUST call this before attempting to
            produce the structured response.

            Do NOT include the actual response data - just call this to signal readiness.
            """
            return f"Ready to generate {schema_name} structured output"

        # Initialize the parent LLMTool with minimal parameters
        super().__init__(
            name=self.TOOL_NAME,
            description=ready_for_structured_output.__doc__.strip(),
            parameters={"type": "object", "properties": {}, "required": []},
            func=ready_for_structured_output,
            strict=False,
        )
        logger.info(f"Initialized ReadyForStructuredOutputTool for schema: {schema_name}")

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the signal (no-op, just returns success).

        Returns:
            Signal confirmation message
        """
        schema_name = self.output_schema.__name__
        logger.info(f"LLM signaled readiness for structured output: {schema_name}")
        return {"status": "ready", "message": f"System will now generate {schema_name} structured output"}
