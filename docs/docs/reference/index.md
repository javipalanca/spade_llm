# API Reference

Complete API documentation for SPADE_LLM components.

## Core Components

- **[Agent](api/agent.md)** - LLMAgent and ChatAgent classes
- **[Behaviour](api/behaviour.md)** - LLMBehaviour implementation  
- **[Providers](api/providers.md)** - LLM provider interfaces
- **[Tools](api/tools.md)** - Tool system and LLMTool class
- **[Memory](api/memory.md)** - Memory system API for agent learning and persistence
- **[Human Interface](api/human-interface.md)** - Human-in-the-loop API and integration
- **[Guardrails](api/guardrails.md)** - Content filtering and safety controls
- **[Context](api/context.md)** - Context and conversation management
- **[Routing](api/routing.md)** - Message routing system

## Quick Reference

### Creating Agents

```python
from spade_llm import LLMAgent, LLMProvider

provider = LLMProvider.create_openai(api_key="key", model="gpt-4o-mini")
agent = LLMAgent(jid="agent@server.com", password="pass", provider=provider)
```

### Creating Tools

```python
from spade_llm import LLMTool

async def my_function(param: str) -> str:
    return f"Result: {param}"

tool = LLMTool(
    name="my_function",
    description="Description of function",
    parameters={"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
    func=my_function
)
```

### Message Routing

```python
def router(msg, response, context):
    if "technical" in response.lower():
        return "tech@example.com"
    return str(msg.sender)

agent = LLMAgent(..., routing_function=router)
```

## Examples

See **[Examples](examples.md)** for complete working code examples.

## Type Definitions

### Common Types

```python
# Message context
ContextMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]

# Routing result
RoutingResult = Union[str, List[str], RoutingResponse, None]

# Tool parameters
ToolParameters = Dict[str, Any]  # JSON Schema format
```

## Error Handling

All SPADE_LLM components use standard Python exceptions:

- `ValueError` - Invalid parameters or configuration
- `ConnectionError` - Network or provider connection issues  
- `TimeoutError` - Operations that exceed timeout limits
- `RuntimeError` - General runtime errors

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your-api-key
OLLAMA_BASE_URL=http://localhost:11434/v1
LM_STUDIO_BASE_URL=http://localhost:1234/v1
```

### Provider Configuration

```python
# OpenAI
provider = LLMProvider.create_openai(api_key="key", model="gpt-4o-mini")

# Ollama  
provider = LLMProvider.create_ollama(model="llama3.1:8b")

# LM Studio
provider = LLMProvider.create_lm_studio(model="local-model")
```

For detailed API documentation, see the individual component pages.
