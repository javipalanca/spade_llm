"""Unified LLM provider implementation for SPADE_LLM."""

import json
import logging
import litellm
from typing import Any, Dict, List, Optional, Sequence

from ..context import ContextManager
from ..tools import LLMTool
from .base_provider import BaseLLMProvider

logger = logging.getLogger("spade_llm.providers")

class LLMProvider(BaseLLMProvider):
    """
    Unified provider for different LLM services with a consistent interface using LiteLLM.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        timeout: float = 600.0,
        max_tokens: Optional[int] = None,
        num_retries: int = 0,
        **kwargs,
    ):
        """
        Initialize the LLM provider with LiteLLM.

        Args:
            model: Model identifier in 'provider/model' format (see supported LLMs: https://docs.litellm.ai/docs/providers/)
            api_key: API key for the provider. Optional for local providers.
            base_url: Custom base URL for the API endpoint (for custom deployments)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            num_retries: Number of retries on failure
            **kwargs: Additional parameters passed to LiteLLM (e.g., fallbacks, 
                             cache, custom_llm_provider, etc.)
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.kwargs = kwargs

        logger.info(f"Initializing provider with model: {self.model}")
        if self.base_url:
            logger.info(f"Using custom base URL: {self.base_url}")

    def _build_completion_kwargs(
        self, 
        messages: Sequence[Any], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Build kwargs for LiteLLM completion call."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "num_retries": self.num_retries,
            **self.kwargs,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        if self.base_url:
            kwargs["api_base"] = self.base_url

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return kwargs

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
        prompt = context.get_prompt(conversation_id)
        logger.info(f"Sending prompt to {self.model}")
        logger.debug(f"Prompt: {prompt}")

        # Prepare tools if they are provided
        formatted_tools = None
        if tools:
            formatted_tools = [tool.to_openai_tool() for tool in tools]
            logger.debug(
                f"Available tools: {[tool['function']['name'] for tool in formatted_tools]}"
            )

        try:
            completion_kwargs = self._build_completion_kwargs(prompt, formatted_tools)

            # Call LiteLLM async completion
            response = await litellm.acompletion(**completion_kwargs)
            message = response.choices[0].message
            result = {"tool_calls": [], "text": None}

            # Process tool calls if present
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.info(
                    f"LLM suggested {len(message.tool_calls)} tool calls"
                )

                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        if isinstance(tc.function.arguments, str):
                            args = json.loads(tc.function.arguments)
                        else:
                            args = tc.function.arguments
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse tool arguments: {tc.function.arguments}, error: {e}"
                        )
                        args = {}

                    tool_call = {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    }
                    tool_calls.append(tool_call)
                    logger.debug(f"Tool call: {tool_call}")

                result["tool_calls"] = tool_calls
            else:
                content = message.content or ""
                if content:
                    logger.info(
                        f"Received text response: {content[:100]}..."
                    )
                else:
                    logger.warning("Received empty response from LLM")
                result["text"] = content

            return result

        except Exception as e:
            logger.error(f"LLM completion error: {e}", exc_info=True)
            raise

    # Legacy methods that delegate to the main method (for backwards compatibility)
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

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to generate embeddings for

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            Exception: If the API call fails
        """
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")

        try:
            # Build embedding kwargs
            embedding_kwargs = {
                "model": self.model,
                "input": texts,
            }

            if self.api_key:
                embedding_kwargs["api_key"] = self.api_key
            
            if self.base_url:
                embedding_kwargs["api_base"] = self.base_url

            # Call LiteLLM async embedding
            response = await litellm.aembedding(**embedding_kwargs)

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response.data]

            logger.debug(
                f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding error: {e}", exc_info=True)
            raise
