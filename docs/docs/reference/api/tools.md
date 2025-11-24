# Tools API

API reference for the SPADE_LLM tools system.

## LLMTool

Core tool class for defining executable functions.

### Constructor

```python
LLMTool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    func: Callable[..., Any]
)
```

**Parameters:**

- `name` - Unique tool identifier
- `description` - Tool description for LLM understanding
- `parameters` - JSON Schema parameter definition
- `func` - Python function to execute

### Methods

#### execute()

```python
async def execute(self, **kwargs) -> Any
```

Execute tool with provided arguments.

**Example:**

```python
result = await tool.execute(city="Madrid", units="celsius")
```

#### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

Convert tool to dictionary representation.

#### to_openai_tool()

```python
def to_openai_tool(self) -> Dict[str, Any]
```

Convert to OpenAI tool format.

### Example

```python
from spade_llm import LLMTool

async def get_weather(city: str, units: str = "celsius") -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

weather_tool = LLMTool(
    name="get_weather",
    description="Get current weather information for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"]
    },
    func=get_weather
)
```

## Parameter Schema

Tools use JSON Schema for parameter validation:

### Basic Types

```python
# String parameter
"city": {
    "type": "string",
    "description": "City name"
}

# Number parameter  
"temperature": {
    "type": "number",
    "minimum": -100,
    "maximum": 100
}

# Boolean parameter
"include_forecast": {
    "type": "boolean",
    "default": False
}

# Array parameter
"cities": {
    "type": "array",
    "items": {"type": "string"},
    "maxItems": 10
}
```

### Complex Schema

```python
parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "filters": {
            "type": "object",
            "properties": {
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"}
                    }
                },
                "category": {
                    "type": "string",
                    "enum": ["news", "blogs", "academic"]
                }
            }
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 10
        }
    },
    "required": ["query"]
}
```

## LangChain Integration

### LangChainToolAdapter

```python
from spade_llm.tools import LangChainToolAdapter
from langchain_community.tools import DuckDuckGoSearchRun

# Create LangChain tool
search_lc = DuckDuckGoSearchRun()

# Adapt for SPADE_LLM
search_tool = LangChainToolAdapter(search_lc)

# Use with agent
agent = LLMAgent(
    jid="agent@example.com",
    password="password",
    provider=provider,
    tools=[search_tool]
)
```

## RetrievalTool

Tool that enables LLM agents to query RetrievalAgent for document retrieval.

### Constructor

```python
RetrievalTool(
    retrieval_agent_jid: str,
    agent_instance: Optional[LLMAgent] = None,
    default_k: int = 4,
    timeout: int = 30,
    name: str = "retrieve_documents",
    description: Optional[str] = None
)
```

**Parameters:**

- `retrieval_agent_jid` - JID of the RetrievalAgent to query
- `agent_instance` - The LLM agent instance (automatically set when added to agent)
- `default_k` - Default number of documents to retrieve (default: 4, range: 1-20)
- `timeout` - Query timeout in seconds (default: 30)
- `name` - Tool name used by LLM (default: "retrieve_documents")
- `description` - Description of what the tool does (uses default if None)

### Usage

```python
from spade_llm import LLMAgent
from spade_llm.tools import RetrievalTool
from spade_llm.providers import LLMProvider

retrieval_tool = RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    default_k=5,
    timeout=30,
    name="knowledge_base",
    description="Search the documentation for information about SPADE-LLM"
)

provider = LLMProvider.create_openai(api_key="your-key")
llm_agent = LLMAgent(
    jid="assistant@localhost",
    password="password",
    provider=provider,
    tools=[retrieval_tool]
)

await llm_agent.start()
```

### How It Works

1. LLM decides to use the retrieval tool
2. Tool sends XMPP message to RetrievalAgent
3. RetrievalAgent performs semantic search
4. Results returned to LLM as context
5. LLM generates response using retrieved information

The LLM can optionally specify `k` and `filters` when calling the tool.

### Recommendations

**Name**: Choose descriptive names that help the LLM understand purpose:
```python
# Good
RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    name="faq_search",
    description="Search frequently asked questions"
)

# Less clear
RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    name="search"
)
```

**Description**: Provide detailed descriptions:
```python
# Good
RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    name="docs_search",
    description="Search the SPADE-LLM documentation for information about agents, behaviors, tools, and configuration. Returns relevant code examples and explanations."
)

# Less helpful
RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    description="Search documentation"
)
```

## Best Practices

### Tool Design

- **Single Purpose**: Each tool should do one thing well
- **Clear Names**: Use descriptive tool names
- **Good Descriptions**: Help LLM understand when to use tools
- **Validate Inputs**: Always validate and sanitize parameters
- **Handle Errors**: Return meaningful error messages
- **Use Async**: Enable concurrent execution
- **RAG Integration**: Use RetrievalTool for knowledge base access
