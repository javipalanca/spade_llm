# Providers API

API reference for LLM provider classes.

## LLMProvider

Unified interface for different LLM services.

### Constructor

```python
LLMProvider(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 1.0,
    timeout: float = 600.0,
    max_tokens: Optional[int] = None,
    num_retries: int = 0,
    **kwargs
) -> LLMProvider
```

Create a provider instance. Uses [LiteLLM model format](https://docs.litellm.ai/docs/providers) for the `model` parameter.

**Parameters:**

- `model` - Model name in LiteLLM format (e.g., `"gpt-4o-mini"`, `"ollama/llama3.1:8b"`, `"openai/local-model"`)
- `api_key` - API key (optional, can be set via environment variables)
- `base_url` - Custom API base URL (for OpenAI-compatible servers)
- `temperature` - Sampling temperature (0.0-2.0)
- `timeout` - Request timeout in seconds
- `max_tokens` - Maximum tokens to generate
- `num_retries` - Number of retries on failure

**Examples:**

```python
# OpenAI
provider = LLMProvider(
    model="gpt-4o-mini",
    api_key="sk-...",
    temperature=0.7,
)

# Ollama (local)
provider = LLMProvider(
    model="ollama/llama3.1:8b",
    timeout=120.0,
)

# OpenAI-compatible API (LM Studio, vLLM, etc.)
provider = LLMProvider(
    model="openai/local-model",
    base_url="http://localhost:1234/v1",
)
```

### Instance Methods

#### get_llm_response()

```python
async def get_llm_response(
    self, 
    context: ContextManager, 
    tools: Optional[List[LLMTool]] = None,
    conversation_id: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]
```

Get complete response from LLM.

**Parameters:**

- `context` - The conversation context manager
- `tools` - Optional list of tools available for this call
- `conversation_id` - Optional conversation ID for multi-conversation contexts
- `output_schema` - Optional Pydantic `BaseModel` class. When provided (and no tools), the response will be parsed into the given schema using the provider's structured output API. When both `output_schema` and tools are present, the [two-phase structured output pattern](../../guides/structured-output.md) is used instead (handled by `LLMBehaviour`).

**Returns:**

```python
{
    'text': Optional[str],       # Text response
    'tool_calls': List[Dict],    # Tool calls requested
    'structured': Optional[Any]  # Parsed Pydantic model instance (when output_schema is used)
}
```

**Example:**

```python
response = await provider.get_llm_response(context, tools)

if response['tool_calls']:
    # Handle tool calls
    for call in response['tool_calls']:
        print(f"Tool: {call['name']}, Args: {call['arguments']}")
else:
    # Handle text response
    print(f"Response: {response['text']}")
```

#### get_response() (Legacy)

```python
async def get_response(
    self, 
    context: ContextManager, 
    tools: Optional[List[LLMTool]] = None
) -> Optional[str]
```

Get text response only.

**Example:**

```python
text_response = await provider.get_response(context)
```

#### get_tool_calls() (Legacy)

```python
async def get_tool_calls(
    self, 
    context: ContextManager, 
    tools: Optional[List[LLMTool]] = None
) -> List[Dict[str, Any]]
```

Get tool calls only.

## BaseProvider

Abstract base class for custom providers.

```python
from spade_llm.providers.base_provider import LLMProvider as BaseProvider

class CustomProvider(BaseProvider):
    async def get_llm_response(self, context, tools=None):
        """Implement custom LLM integration."""
        # Your implementation
        return {
            'text': "Response from custom provider",
            'tool_calls': []
        }
```

## Provider Configuration

### Model Format

SPADE-LLM uses [LiteLLM's model naming convention](https://docs.litellm.ai/docs/providers):

```
provider_prefix/model_name
```

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `model_name` (no prefix) | `gpt-4o-mini` |
| Ollama | `ollama/model_name` | `ollama/llama3.1:8b` |
| Anthropic | `anthropic/model_name` | `anthropic/claude-3-5-sonnet` |
| OpenAI-compat | `openai/model_name` + `base_url` | `openai/local-model` |



## Embeddings

### get_embeddings()

Generate embeddings for RAG systems.

```python
async def get_embeddings(texts: List[str]) -> List[List[float]]
```

**Parameters:**

- `texts` - List of text strings to embed

**Returns:**

- List of embedding vectors (each vector is a list of floats)
