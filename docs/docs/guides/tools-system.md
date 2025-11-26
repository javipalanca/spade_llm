# Tools System

Enable LLM agents to **execute functions** and **interact with external services**.

## Tool Calling Flow

```mermaid
flowchart TD
    A[User sends message] --> B{Does LLM need tools?}
    B -->|No| C[Generate direct response]
    B -->|Yes| D[Decide which tool to use]
    D --> E[Determine required arguments]
    E --> F[Send tool_calls in response]
    F --> G[System executes tool]
    G --> H[Add results to context]
    H --> I[Second LLM query with results]
    I --> J[Generate final response]
    C --> K[Send response to user]
    J --> K
```

## Overview

The **Tools System** empowers LLM agents to _extend beyond conversation_ by executing real functions. This enables agents to:

- **🔧 Execute Python functions** with dynamic parameters
- **🌐 Access external APIs** and databases  
- **📁 Process files** and perform calculations
- **🔗 Integrate with third-party services**

## How Tool Calling Works

When an LLM agent receives a message, it can either respond directly or **decide to use tools**. The process involves:

1. **_Intelligence Decision_**: The LLM analyzes if it needs external data or functionality
2. **_Tool Selection_**: It chooses the appropriate tool from available options
3. **_Parameter Generation_**: The LLM determines what arguments the tool needs
4. **_Execution_**: The system runs the tool function asynchronously
5. **_Context Integration_**: Results are added back to the conversation
6. **_Final Response_**: The LLM processes results and provides a complete answer

## Basic Tool Definition

```python
from spade_llm import LLMTool

async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

weather_tool = LLMTool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    },
    func=get_weather
)
```

## Using Tools with Agents

```python
from spade_llm import LLMAgent, LLMProvider

agent = LLMAgent(
    jid="assistant@example.com",
    password="password",
    provider=provider,
    tools=[weather_tool]  # Register tools
)
```

When the LLM needs weather information, it will **automatically detect the need** and call the tool.

## Common Tool Categories

### **🌐 API Integration**
Connect to external web services for real-time data.

```python
import aiohttp

async def web_search(query: str) -> str:
    """Search the web for information."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.duckduckgo.com/?q={query}&format=json") as response:
            data = await response.json()
            return str(data)

search_tool = LLMTool(
    name="web_search",
    description="Search the web for current information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    },
    func=web_search
)
```

### **📁 File Operations**
Read, write, and process files on the system.

```python
import aiofiles

async def read_file(filepath: str) -> str:
    """Read a text file."""
    try:
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
        return f"File content:\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"

file_tool = LLMTool(
    name="read_file",
    description="Read contents of a text file",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string"}
        },
        "required": ["filepath"]
    },
    func=read_file
)
```

### **📊 Data Processing**
Perform calculations and data analysis.

```python
import json

async def calculate_stats(numbers: list) -> str:
    """Calculate statistics for a list of numbers."""
    if not numbers:
        return "Error: No numbers provided"
    
    stats = {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }
    return json.dumps(stats, indent=2)

stats_tool = LLMTool(
    name="calculate_stats",
    description="Calculate basic statistics",
    parameters={
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "required": ["numbers"]
    },
    func=calculate_stats
)
```

### **🧠 Human Expert Consultation**
Connect LLM agents with human experts for real-time guidance and decision support.

```python
from spade_llm.tools import HumanInTheLoopTool

# Create human expert consultation tool
human_expert = HumanInTheLoopTool(
    human_expert_jid="expert@company.com",
    timeout=300.0,  # 5 minutes
    name="ask_human_expert",
    description="""Ask a human expert for help when you need:
    - Current information not in your training data
    - Human judgment or subjective opinions
    - Company-specific policies or procedures
    - Clarification on ambiguous requests"""
)

# Use with agent
agent = LLMAgent(
    jid="assistant@company.com",
    password="password",
    provider=provider,
    tools=[human_expert],
    system_prompt="""You are an AI assistant with access to human experts.
    When you encounter questions requiring human judgment, current information,
    or company-specific knowledge, consult the human expert."""
)
```

**Key Features:**

- **⚡ Real-time consultation** via XMPP messaging
- **🌐 Web interface** for human experts to respond
- **🔄 Message correlation** using XMPP thread IDs
- **⏱️ Configurable timeouts** with graceful error handling
- **🔒 Template-based filtering** prevents message conflicts

**When the LLM uses this tool:**

1. **Question sent** to human expert via XMPP
2. **Expert receives** notification in web interface  
3. **Human provides** response through browser
4. **Response returns** to LLM via XMPP
5. **Agent continues** with human-informed answer

**Example consultation flow:**
```
User: "What's our company policy on remote work?"
Agent: [Uses ask_human_expert tool]
→ Human Expert: "We allow 3 days remote per week with manager approval"
Agent: "According to our HR expert, our policy allows up to 3 days 
       remote work per week with manager approval."
```

!!! tip "Setup Required"
    Human-in-the-loop requires XMPP server with WebSocket support and web interface.
    See the [Human-in-the-Loop guide](../guides/human-in-the-loop.md) and [Examples documentation](../reference/examples.md) for complete setup instructions.

## **📚 RAG Integration - RetrievalTool**

Enable LLM agents to **query knowledge bases** using RetrievalTool.

### What is RetrievalTool?

RetrievalTool allows LLM agents to search document collections via a dedicated RetrievalAgent, enabling **Retrieval-Augmented Generation (RAG)**.

### Setup

```python
from spade_llm import LLMAgent, RetrievalAgent
from spade_llm.tools import RetrievalTool
from spade_llm.providers import LLMProvider
from spade_llm.rag import Chroma, VectorStoreRetriever

# 1. Create vector store and retriever
embedding_provider = LLMProvider.create_ollama(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="documentation",
    embedding_fn=embedding_provider.get_embeddings
)
await vector_store.initialize()
# ... add documents to vector store ...

retriever = VectorStoreRetriever(vector_store=vector_store)

# 2. Create retrieval agent
retrieval_agent = RetrievalAgent(
    jid="retrieval@localhost",
    password="retrieval_pass",
    retriever=retriever
)
await retrieval_agent.start()

# 3. Create retrieval tool
retrieval_tool = RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    default_k=5,  # Retrieve top 5 documents by default
    name="knowledge_base",
    description="Search documentation for information about SPADE-LLM, agents, tools, and configuration"
)

# 4. Add to LLM agent
llm_provider = LLMProvider.create_openai(api_key="your-key")

llm_agent = LLMAgent(
    jid="assistant@localhost",
    password="assistant_pass",
    llm_provider=llm_provider,
    tools=[retrieval_tool]
)
await llm_agent.start()
```

### How It Works

```mermaid
sequenceDiagram
    participant U as User
    participant L as LLM Agent
    participant R as RetrievalTool
    participant RA as RetrievalAgent
    participant VS as Vector Store

    U->>L: "How do I create a tool?"
    L->>L: Decides to use knowledge_base
    L->>R: Execute tool with query
    R->>RA: XMPP message
    RA->>VS: Semantic search
    VS->>RA: Relevant documents
    RA->>R: Results
    R->>L: Documents as context
    L->>U: Answer with context
```

### Multi-Agent RAG Pattern

Multiple LLM agents can share a single RetrievalAgent:

```python
# One retrieval agent
retrieval_agent = RetrievalAgent(
    jid="retrieval@localhost",
    password="retrieval_pass",
    retriever=retriever
)

# Multiple LLM agents with the same tool
assistant_1 = LLMAgent(
    jid="assistant1@localhost",
    password="pass1",
    llm_provider=provider1,
    tools=[RetrievalTool(
        retrieval_agent_jid="retrieval@localhost",
        name="kb",
        description="Search docs"
    )]
)

assistant_2 = LLMAgent(
    jid="assistant2@localhost",
    password="pass2",
    llm_provider=provider2,
    tools=[RetrievalTool(
        retrieval_agent_jid="retrieval@localhost",
        name="kb",
        description="Search docs"
    )]
)
```

### Configuration Tips

**Name and Description**: Be specific to help LLM understand when to use the tool:

```python
# Good
RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    name="technical_docs",
    description="Search technical documentation for code examples, API references, and implementation guides"
)

# Less helpful
RetrievalTool(
    retrieval_agent_jid="retrieval@localhost",
    name="search",
    description="Search"
)
```

### Benefits

- **Up-to-date information**: Access current data without retraining
- **Domain expertise**: Query specialized knowledge bases
- **Verifiable**: Trace answers to source documents
- **Reduced hallucinations**: Ground responses in actual documents
- **Scalable**: Separate retrieval concerns from LLM logic

See **[RAG System Guide](rag-system.md)** for complete RAG documentation.

## **LangChain Integration**

_Seamlessly use existing LangChain tools_ with SPADE_LLM:

```python
from langchain_community.tools import DuckDuckGoSearchRun
from spade_llm.tools import LangChainToolAdapter

# Create LangChain tool
search_lc = DuckDuckGoSearchRun()

# Adapt for SPADE_LLM  
search_tool = LangChainToolAdapter(search_lc)

# Use with agent
agent = LLMAgent(
    jid="assistant@example.com",
    password="password", 
    provider=provider,
    tools=[search_tool]
)
```


## **✅ Best Practices**

- **_Single Purpose_**: Each tool should do **one thing well**
- **_Clear Naming_**: Use **descriptive tool names** that explain functionality
- **_Rich Descriptions_**: Help the LLM understand **when and how** to use tools
- **_Input Validation_**: Always **validate and sanitize** inputs for security
- **_Meaningful Errors_**: Return **clear error messages** for troubleshooting
- **_Async Functions_**: Use **async/await** for non-blocking execution
- **_RAG for Knowledge_**: Use RetrievalTool for document-based information


## Next Steps

- **[RAG System](rag-system.md)** - Complete RAG implementation guide
- **[MCP Integration](mcp.md)** - Connect to external MCP servers
- **[Architecture](architecture.md)** - Understanding system design
- **[Providers](providers.md)** - LLM provider configuration
