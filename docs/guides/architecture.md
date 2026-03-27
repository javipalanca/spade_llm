# Architecture

SPADE_LLM **extends SPADE's multi-agent framework** with LLM capabilities while maintaining _full compatibility_.

## Component Overview

```mermaid
graph TB
    A[LLMAgent] --> B[LLMBehaviour]
    B --> C[ContextManager]
    B --> D[LLMProvider]
    B --> E[LLMTool]
    B --> I[Guardrails System]
    
    D --> F[OpenAI/Ollama/etc]
    E --> G[Python Functions]
    E --> H[MCP Servers]
    I --> J[Input Filters]
    I --> K[Output Filters]
```

## **🏗️ Core Components**

### **🤖 LLMAgent**
The **main agent class** that extends SPADE's `Agent` with LLM capabilities:

- **Manages LLM provider connection** and configuration
- **Registers tools** and handles their lifecycle
- **Controls conversation limits** and termination conditions
- Provides the _bridge_ between SPADE's XMPP messaging and LLM processing

### **⚡ LLMBehaviour**  
The **core processing engine** that orchestrates the entire LLM workflow:

1. **Receives XMPP messages** from other agents
2. **Updates conversation context** with new information
3. **Calls LLM provider** for responses
4. **Executes tools** when requested by the LLM
5. **Routes responses** to appropriate recipients

This is where the main processing occurs - transforming simple messages into interactions.

### **🧠 ContextManager**
**Manages conversation state** across multiple concurrent discussions:

- **Tracks multiple conversations** simultaneously by thread ID
- **Formats messages** appropriately for different LLM providers
- **Handles context windowing** to manage token limits efficiently
- Ensures each conversation maintains its _own context_ and history

### **🔌 LLMProvider**
**Unified interface** for different LLM services, providing consistency:

- **Abstracts provider-specific APIs** (OpenAI, Ollama, Anthropic, etc.)
- **Handles tool calling formats** across different providers
- **Provides consistent error handling** and retry mechanisms
- Makes it _easy to switch_ between different LLM services

### **🛠️ LLMTool**
**Framework for executable functions** that extend LLM capabilities:

- **Async execution support** for non-blocking operations
- **JSON Schema parameter validation** for type safety
- **Integration with LangChain and MCP** for ecosystem compatibility
- Enables LLMs to perform _real actions_ beyond conversation

## **📨 Message Flow**

```mermaid
sequenceDiagram
    participant A as External Agent
    participant B as LLMBehaviour
    participant C as LLMProvider
    participant D as LLM Service
    participant E as LLMTool

    A->>B: XMPP Message
    B->>C: Get Response
    C->>D: API Call
    D->>C: Tool Calls
    C->>B: Tool Requests
    loop For Each Tool
        B->>E: Execute
        E->>B: Result
    end
    B->>C: Get Final Response
    C->>D: API Call
    D->>C: Final Response
    B->>A: Response Message
```

## **🔄 Conversation Lifecycle**

The conversation lifecycle follows a **well-defined process**:

1. **Initialization**: New conversation created from message thread
2. **Processing**: Messages processed through LLM with tool execution
3. **Termination**: Ends via markers, limits, or manual control
4. **Cleanup**: Resources freed and callbacks executed

Each stage ensures conversations can handle complex, multi-turn interactions while maintaining system stability.



## **🧩 RAG System Architecture**

SPADE-LLM extends the multi-agent framework with **Retrieval-Augmented Generation (RAG)** capabilities, allowing agents to query external knowledge bases:

```mermaid
graph LR
    A[LLM Agent] --> B{Needs Info?}
    B -->|Yes| C[RetrievalTool]
    C --> D[RetrievalAgent]
    D --> E[Vector Store]
    E --> D
    D --> C
    C --> A
    B -->|No| F[Direct Response]
    A --> F
    
    style A fill:#FF9800
    style D fill:#2196F3
    style E fill:#4CAF50
```

### **How It Works**

RAG in SPADE-LLM operates through **agent collaboration**:

1. **RetrievalAgent** manages a knowledge base (documents stored as vectors in ChromaDB)
2. **LLM agents** query it via **RetrievalTool** using standard XMPP messaging
3. Retrieved documents are automatically added to the LLM's context
4. Responses are grounded in actual source material, not just model memory

### **Key Advantages**

- **Multi-agent RAG**: Deploy multiple retrieval agents for different domains (HR docs, technical docs, etc.)
- **Distributed knowledge**: Each department can run its own retrieval service
- **Universal compatibility**: Works with any SPADE agent, not just LLM-based ones
- **Autonomous decision-making**: LLMs decide when to retrieve information vs. using existing knowledge

### **Example**

```python
tech_docs_agent = RetrievalAgent("tech_docs@localhost", retriever=tech_retriever)
hr_docs_agent = RetrievalAgent("hr_docs@localhost", retriever=hr_retriever)

assistant = LLMAgent(
    "assistant@localhost",
    provider=llm_provider,
    tools=[
        RetrievalTool("tech_docs", "Technical documentation", "tech_docs@localhost"),
        RetrievalTool("hr_docs", "HR policies", "hr_docs@localhost")
    ]
)
```

For detailed implementation guidance, see the **[RAG System Guide](rag-system.md)**.

## **🔧 Integration Points**

The architecture provides multiple **integration points** for customization:

- **Custom Providers**: Add new LLM services
- **Tool Extensions**: Create domain-specific tools
- **Routing Logic**: Implement custom message routing
- **Context Management**: Customize conversation handling
- **MCP Integration**: Connect to external servers
- **RAG Components**: Custom document loaders, splitters, and retrievers
- **Embedding Models**: Use different embedding providers

This **flexible design** ensures SPADE_LLM can adapt to various use cases while maintaining its core multi-agent capabilities.

## Next Steps

- **[Providers](providers.md)** - Configure LLM providers
- **[Tools System](tools-system.md)** - Add tool capabilities
- **[RAG System](rag-system.md)** - Implement retrieval-augmented generation
- **[Routing](routing.md)** - Implement message routing
- **[MCP](mcp.md)** - Connect to external services
