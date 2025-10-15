"""Unified LLM provider implementation for SPADE_LLM."""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError

from ..context import ContextManager
from ..tools import LLMTool

logger = logging.getLogger("spade_llm.providers")


class ModelFormat(Enum):
    """Supported model format conventions."""

    OPENAI = "openai"  # Standard OpenAI format (e.g., "gpt-4")
    OLLAMA = "ollama"  # Ollama format (e.g., "llama3:latest")
    CUSTOM = "custom"  # Custom format (e.g., "custom/model-name")


class LLMProvider:
    """
    Unified provider for different LLM services with a consistent interface.

    This class abstracts the differences between various LLM APIs (OpenAI, Ollama, etc.)
    and provides a consistent interface for SPADE agents to interact with LLMs.
    """

    def __init__(
        self,
        api_key: str = "dummy",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_format: Optional[ModelFormat] = None,
        provider_name: Optional[str] = None,
    ):
        """
        Initialize the LLM provider.

        Note: Users should not call this constructor directly.
        Use the create_* class methods instead.

        Args:
            api_key: API key. For local models without auth, can be any string.
            model: The model to use. Format depends on the service.
            temperature: Temperature for generation (0.0 to 1.0).
            base_url: Base URL for the API endpoint.
            timeout: Timeout in seconds for API calls.
            max_tokens: Maximum tokens to generate.
            model_format: Format convention for model names.
            provider_name: Name of the provider for logging purposes.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout or 60.0
        self.max_tokens = max_tokens
        self.model_format = model_format or self._detect_model_format(model, base_url)
        self.provider_name = provider_name or self._detect_provider_name(
            base_url, self.model_format
        )

        # Initialize client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        logger.info(
            f"Initializing {self.provider_name} provider with model: {self.model}"
        )
        if self.base_url:
            logger.info(f"Using base URL: {self.base_url}")

        self.client = OpenAI(**client_kwargs)

    def _detect_model_format(self, model: str, base_url: Optional[str]) -> ModelFormat:
        """Detect the model format based on the model name and base URL."""
        if model.startswith("ollama/"):
            return ModelFormat.OLLAMA
        elif model.startswith("gpt-") or model.startswith("o1-"):
            return ModelFormat.OPENAI
        elif "/" in model:
            return ModelFormat.CUSTOM
        else:
            # Default based on base_url
            if base_url and "ollama" in base_url.lower():
                return ModelFormat.OLLAMA
            return ModelFormat.OPENAI

    def _detect_provider_name(
        self, base_url: Optional[str], model_format: ModelFormat
    ) -> str:
        """Detect the provider name for logging."""
        if not base_url:
            return "OpenAI"

        url_lower = base_url.lower()
        if "ollama" in url_lower:
            return "Ollama"
        elif "vllm" in url_lower:
            return "vLLM"
        elif "lmstudio" in url_lower:
            return "LM Studio"
        elif "localhost" in url_lower or "127.0.0.1" in url_lower:
            return "Local OpenAI-compatible"
        else:
            return "OpenAI-compatible"

    def _prepare_model_name(self, model: str) -> str:
        """Prepare the model name for the API call based on the format."""
        if self.model_format == ModelFormat.OLLAMA and model.startswith("ollama/"):
            # Remove the "ollama/" prefix for Ollama API
            return model[7:]
        return model

    @classmethod
    def create_openai(
        cls,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **kwargs,
    ) -> "LLMProvider":
        """
        Create a provider configured for OpenAI API.

        Args:
            api_key: OpenAI API key
            model: Model name to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional model parameters

        Returns:
            Configured LLMProvider instance
        """
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider_name="OpenAI",
            **kwargs,
        )

    @classmethod
    def create_ollama(
        cls,
        model: str = "llama3:1b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        timeout: float = 120.0,
        **kwargs,
    ) -> "LLMProvider":
        """
        Create a provider configured for Ollama API.

        Args:
            model: Model name to use (e.g., "llama3:8b", "gemma:2b")
            base_url: URL for the Ollama API (must include /v1)
            temperature: Sampling temperature (0.0 to 1.0)
            timeout: Timeout in seconds for API calls
            **kwargs: Additional model parameters

        Returns:
            Configured LLMProvider instance
        """
        # Add ollama/ prefix if not present
        if not model.startswith("ollama/"):
            model = f"ollama/{model}"

        return cls(
            api_key="dummy",
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            provider_name="Ollama",
            model_format=ModelFormat.OLLAMA,
            **kwargs,
        )

    @classmethod
    def create_lm_studio(
        cls,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.7,
        **kwargs,
    ) -> "LLMProvider":
        """
        Create a provider configured for LM Studio.

        Args:
            model: Model name as defined in LM Studio
            base_url: URL for the LM Studio API
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional model parameters

        Returns:
            Configured LLMProvider instance
        """
        return cls(
            api_key="dummy",
            model=model,
            base_url=base_url,
            temperature=temperature,
            provider_name="LM Studio",
            **kwargs,
        )

    @classmethod
    def create_vllm(
        cls, model: str, base_url: str = "http://localhost:8000/v1", **kwargs
    ) -> "LLMProvider":
        """
        Create a provider configured for vLLM.

        Args:
            model: Model name to use
            base_url: URL for the vLLM API
            **kwargs: Additional model parameters

        Returns:
            Configured LLMProvider instance
        """
        return cls(
            api_key="dummy",
            model=model,
            base_url=base_url,
            provider_name="vLLM",
            **kwargs,
        )

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
        logger.info(f"Sending prompt to {self.provider_name} ({self.model})")
        logger.debug(f"Prompt: {prompt}")

        # Prepare tools if they are provided
        formatted_tools = None
        if tools:
            formatted_tools = [tool.to_openai_tool() for tool in tools]
            logger.debug(
                f"Available tools: {[tool['function']['name'] for tool in formatted_tools]}"
            )

        try:
            # Prepare the completion kwargs
            completion_kwargs = {
                "model": self._prepare_model_name(self.model),
                "messages": prompt,
                "temperature": self.temperature,
                "timeout": self.timeout,
            }

            # Add optional parameters
            if self.max_tokens:
                completion_kwargs["max_tokens"] = self.max_tokens

            if formatted_tools:
                completion_kwargs["tools"] = formatted_tools
                completion_kwargs["tool_choice"] = "auto"

                # Note for Ollama users
                if self.provider_name == "Ollama":
                    logger.info(
                        "Using Ollama with tool support. Ensure you're using a tool-capable model (e.g., Llama 3.1+)"
                    )

            response = await asyncio.to_thread(
                self.client.chat.completions.create, **completion_kwargs
            )

            message = response.choices[0].message
            result = {"tool_calls": [], "text": None}

            # Process tool calls if present
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.info(
                    f"{self.provider_name} suggested {len(message.tool_calls)} tool calls"
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
                        f"Received text response from {self.provider_name}: {content[:100]}..."
                    )
                else:
                    logger.warning(f"Received empty response from {self.provider_name}")
                result["text"] = content

            return result

        except OpenAIError as e:
            logger.error(f"OpenAI API error with {self.provider_name}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error with {self.provider_name}: {e}", exc_info=True
            )
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

        This method supports both OpenAI and Ollama embedding models.
        For Ollama, ensure you're using an embedding-capable model
        (e.g., nomic-embed-text, mxbai-embed-large).

        Args:
            texts: List of strings to generate embeddings for

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            OpenAIError: If the API call fails
            ValueError: If the response format is unexpected
        """
        logger.info(
            f"Generating embeddings for {len(texts)} texts using {self.provider_name}"
        )

        try:
            model_name = self._prepare_model_name(self.model)

            # Call the embeddings endpoint
            response = await asyncio.to_thread(
                self.client.embeddings.create, model=model_name, input=texts
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]

            logger.debug(
                f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except OpenAIError as e:
            logger.error(f"OpenAI API error generating embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}", exc_info=True)
            raise
