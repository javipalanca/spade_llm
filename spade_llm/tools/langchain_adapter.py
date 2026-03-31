"""LangChain tools adapter for SPADE_LLM."""

import asyncio
import logging
from typing import Any, Dict

from .llm_tool import LLMTool

logger = logging.getLogger("spade_llm.tools.langchain")


class LangChainToolAdapter(LLMTool):
    """
    Adapter for using LangChain tools within SPADE_LLM.

    This adapter wraps LangChain tools to make them compatible with the
    SPADE_LLM tooling system, allowing seamless integration of the
    extensive LangChain tool ecosystem.
    """

    def __init__(self, langchain_tool):
        """
        Initialize a new LangChain tool adapter.

        Args:
            langchain_tool: A LangChain tool instance.
        """
        self.lc_tool = langchain_tool

        # Extract information from the LangChain tool
        name = getattr(langchain_tool, "name", "unknown_tool")
        description = getattr(langchain_tool, "description", "No description available for this tool.")

        # Convert the parameters schema if it exists
        parameters = self._convert_parameters_schema(langchain_tool)

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            func=self._execute_langchain_tool,
        )

        # Store expected parameter format
        self._expected_param = self._get_expected_parameter_name()

    def _convert_parameters_schema(self, tool) -> Dict[str, Any]:
        """
        Convert LangChain tool parameters schema to OpenAI format.

        Args:
            tool: The LangChain tool.

        Returns:
            A JSON Schema object compatible with OpenAI's function calling.
        """
        # If the tool has a schema, use it
        if hasattr(tool, "args_schema"):
            try:
                # Handle both pydantic v1 and v2
                if hasattr(tool.args_schema, "schema"):
                    # Pydantic v1
                    schema = tool.args_schema.schema()

                    # Ensure it's a proper JSON Schema object
                    if "properties" not in schema:
                        schema = {
                            "type": "object",
                            "properties": schema,
                            "required": [],
                        }
                    return schema
                else:
                    # For pydantic v2 or other schemas
                    logger.warning(f"Unsupported schema type for tool {tool.name}. Using default schema.")
            except Exception as e:
                logger.error(f"Error converting schema for tool {tool.name}: {e}")

        # Default schema for tools without a defined schema
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": f"Input for the {tool.name} tool",
                }
            },
            "required": ["input"],
        }

    def _get_expected_parameter_name(self) -> str:
        """
        Determine the expected parameter name for the LangChain tool.

        Many LangChain tools expect a single parameter named 'input',
        but this can be customized by looking at the schema.

        Returns:
            The name of the main input parameter.
        """
        # Try to determine from schema properties
        if hasattr(self.lc_tool, "args_schema"):
            # For newer LangChain tools, check the schema
            if hasattr(self.lc_tool.args_schema, "schema"):
                schema = self.lc_tool.args_schema.schema()
                if "properties" in schema and len(schema["properties"]) == 1:
                    # If there's only one property, use its name
                    return next(iter(schema["properties"]))

        # Default to 'input' for standard LangChain tools
        return "input"

    async def _execute_langchain_tool(self, **kwargs) -> Any:
        """
        Execute the LangChain tool.

        Args:
            **kwargs: Arguments to pass to the LangChain tool.

        Returns:
            The result of the tool execution.
        """
        try:
            # Convert OpenAI-style parameters to LangChain format
            lc_args = self._transform_parameters(**kwargs)

            logger.debug(f"Executing {self.name} with transformed params: {lc_args}")

            # Determine if the tool is async
            if hasattr(self.lc_tool, "_acall") or hasattr(self.lc_tool, "acall"):
                acall = getattr(self.lc_tool, "_acall", None) or getattr(self.lc_tool, "acall")
                return await acall(**lc_args)
            elif hasattr(self.lc_tool, "_call") or hasattr(self.lc_tool, "call"):
                call = getattr(self.lc_tool, "_call", None) or getattr(self.lc_tool, "call")
                # Execute in a thread to avoid blocking
                return await asyncio.to_thread(call, **lc_args)
            elif hasattr(self.lc_tool, "run") or hasattr(self.lc_tool, "__call__"):
                # Try direct invocation as a last resort
                func = getattr(self.lc_tool, "run", None) or self.lc_tool

                # Check if it's a direct string input tool (most common case)
                if len(kwargs) == 1 and self._expected_param in lc_args:
                    # For tools that expect a direct string argument
                    return await asyncio.to_thread(func, lc_args[self._expected_param])
                else:
                    return await asyncio.to_thread(func, **lc_args)
        except Exception as e:
            logger.error(f"Error executing LangChain tool {self.name}: {e}")
            return {"error": str(e)}

    def _transform_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Transform parameters from OpenAI format to LangChain format.

        Args:
            **kwargs: Parameters in OpenAI format.

        Returns:
            Parameters in LangChain format.
        """
        # For simple tools expecting just one parameter
        if len(kwargs) == 1 and len(self.parameters.get("properties", {})) == 1:
            return {self._expected_param: next(iter(kwargs.values()))}

        # Common parameter names that usually map to the expected parameter
        common_input_params = ["query", "question", "search", "text", "input", "q"]

        # Check if any of the provided parameters need mapping to expected parameter
        for param_name, value in kwargs.items():
            if param_name in common_input_params:
                # Create a new dict with the mapped parameter
                result = {self._expected_param: value}
                # Add any remaining parameters as-is
                for k, v in kwargs.items():
                    if k != param_name:
                        result[k] = v
                return result

        # Try to infer from schema if both schema and kwargs contain the same properties
        # but with different names
        if hasattr(self.lc_tool, "args_schema") and hasattr(self.lc_tool.args_schema, "schema"):
            schema = self.lc_tool.args_schema.schema()
            if "properties" in schema and len(schema["properties"]) == len(kwargs):
                # Try to match parameters by position assuming they're ordered similarly
                schema_props = list(schema["properties"].keys())
                kwargs_props = list(kwargs.keys())

                if len(schema_props) == len(kwargs_props):
                    result = {}
                    for i, (schema_prop, kwarg_prop) in enumerate(zip(schema_props, kwargs_props)):
                        if schema_prop != kwarg_prop:
                            result[schema_prop] = kwargs[kwarg_prop]
                        else:
                            result[schema_prop] = kwargs[schema_prop]
                    return result

        # Return as-is if no mapping is found
        return kwargs
