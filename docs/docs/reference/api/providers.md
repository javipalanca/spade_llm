# Providers API

API reference for LLM provider classes.

## LLMProvider

Unified interface for different LLM services.

### Class Methods

#### create_openai()

```python
LLMProvider.create_openai(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> LLMProvider
```

Create OpenAI provider.

**Parameters:**

- `api_key` - OpenAI API key
- `model` - Model name (e.g., "gpt-4o", "gpt-4o-mini")
- `temperature` - Sampling temperature (0.0-1.0)
- `timeout` - Request timeout in seconds
- `max_tokens` - Maximum tokens to generate

**Example:**

```python
provider = LLMProvider.create_openai(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.7
)
```

#### create_ollama()

```python
LLMProvider.create_ollama(
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434/v1",
    temperature: float = 0.7,
    timeout: float = 120.0
) -> LLMProvider
```

Create Ollama provider.

**Parameters:**

- `model` - Model name (e.g., "llama3.1:8b", "mistral:7b")
- `base_url` - Ollama server URL
- `temperature` - Sampling temperature
- `timeout` - Request timeout (longer for local models)

**Example:**

```python
provider = LLMProvider.create_ollama(
    model="llama3.1:8b",
    temperature=0.8,
    timeout=180.0
)
```

#### create_lm_studio()

```python
LLMProvider.create_lm_studio(
    model: str = "local-model",
    base_url: str = "http://localhost:1234/v1",
    temperature: float = 0.7
) -> LLMProvider
```

Create LM Studio provider.

**Example:**

```python
provider = LLMProvider.create_lm_studio(
    model="Meta-Llama-3.1-8B-Instruct",
    base_url="http://localhost:1234/v1"
)
```

#### create_vllm()

```python
LLMProvider.create_vllm(
    model: str,
    base_url: str = "http://localhost:8000/v1"
) -> LLMProvider
```

Create vLLM provider.

**Example:**

```python
provider = LLMProvider.create_vllm(
    model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000/v1"
)
```

### Instance Methods

#### get_llm_response()

```python
async def get_llm_response(
    self, 
    context: ContextManager, 
    tools: Optional[List[LLMTool]] = None
) -> Dict[str, Any]
```

Get complete response from LLM.

**Returns:**

```python
{
    'text': Optional[str],      # Text response
    'tool_calls': List[Dict]    # Tool calls requested
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

### Model Formats

```python
class ModelFormat(Enum):
    OPENAI = "openai"    # gpt-4, gpt-3.5-turbo
    OLLAMA = "ollama"    # llama3.1:8b, mistral:7b  
    CUSTOM = "custom"    # custom/model-name
```

### Environment Variables

```python
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Ollama  
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1:8b

# LM Studio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=local-model
```

### Dynamic Configuration

```python
import os

def create_provider_from_env():
    provider_type = os.getenv('LLM_PROVIDER', 'openai')
    
    if provider_type == 'openai':
        return LLMProvider.create_openai(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        )
    elif provider_type == 'ollama':
        return LLMProvider.create_ollama(
            model=os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        )

provider = create_provider_from_env()
```

## Tool Support

### OpenAI Tools

Native tool calling support:

```python
# Tools automatically formatted for OpenAI
response = await provider.get_llm_response(context, tools)
```

### Ollama Tools  

Limited to compatible models:

```python
# Check model compatibility
tool_compatible_models = [
    "llama3.1:8b", "llama3.1:70b", "mistral:7b"
]

if model in tool_compatible_models:
    # Use tools
    response = await provider.get_llm_response(context, tools)
```

## Error Handling

```python
from openai import OpenAIError

try:
    response = await provider.get_llm_response(context)
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
except ConnectionError as e:
    print(f"Connection error: {e}")
except TimeoutError as e:
    print(f"Request timeout: {e}")
```

## Provider Comparison

| Feature | OpenAI | Ollama | LM Studio | vLLM |
|---------|--------|--------|-----------|------|
| **Setup** | Easy | Medium | Easy | Hard |
| **Quality** | Excellent | Good | Good | Good |
| **Speed** | Fast | Slow | Slow | Fast |
| **Cost** | Paid | Free | Free | Free |
| **Privacy** | Low | High | High | High |
| **Tools** | Full | Limited | Limited | Limited |

## Best Practices

### Provider Selection

```python
def choose_provider(use_case: str):
    """Choose provider based on use case."""
    if use_case == "development":
        return LLMProvider.create_ollama(model="llama3.1:1b")  # Fast
    elif use_case == "production":
        return LLMProvider.create_openai(model="gpt-4o-mini")   # Reliable
    elif use_case == "privacy":
        return LLMProvider.create_ollama(model="llama3.1:8b")  # Local
```

### Error Recovery

```python
async def robust_llm_call(providers: List[LLMProvider], context):
    """Try multiple providers with fallback."""
    for provider in providers:
        try:
            return await provider.get_llm_response(context)
        except Exception as e:
            print(f"Provider failed: {e}")
            continue
    
    raise Exception("All providers failed")
```

### Performance Monitoring

```python
import time

async def timed_call(provider, context):
    """Monitor provider performance."""
    start = time.time()
    try:
        response = await provider.get_llm_response(context)
        duration = time.time() - start
        print(f"Provider response time: {duration:.2f}s")
        return response
    except Exception as e:
        duration = time.time() - start
        print(f"Provider failed after {duration:.2f}s: {e}")
        raise
```

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
