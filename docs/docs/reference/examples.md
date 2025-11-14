# Examples

Complete working examples for SPADE_LLM applications.

## Repository Examples

The [examples](https://github.com/sosanzma/spade_llm/tree/main/examples) directory contains complete working examples:

**Tool Examples**:
- `ollama_with_tools_example.py` - Local models with tools
- `langchain_tools_example.py` - LangChain tool integration

**Multi-Agent Examples**:
- `multi_provider_chat_example.py` - Chat with different providers
- `simple_coordinator_example.py` - Coordinator agent pattern
- `valencia_smartCity_mcp_example.py` - Smart city multi-agent system

**RAG Examples**:
- `rag_system_ollama_chroma_example.py` - Complete RAG demonstration
- `rag_vs_no_rag_demo.py` - Results comparison with and without RAG

**Specialized Examples**:
- `spanish_to_english_translator.py` - Translation agent
- `trip_planner_example.py` - Trip planning workflow
- `github_issues_monitor_complex_example.py` - GitHub monitoring
- `guardrails_example.py` - Content filtering

## Human-in-the-Loop Example

Complete example demonstrating LLM agents consulting with human experts:

```python
"""
Example: LLM Agent with Human Expert Consultation

This example shows how to create an agent that can ask human experts
for help when it needs current information or human judgment.

Prerequisites:
1. XMPP server with WebSocket support (e.g., OpenFire)
2. Human expert web interface running
3. XMPP accounts for agent, expert, and chat user
"""

import asyncio
import logging
import spade
from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.tools import HumanInTheLoopTool
from spade_llm.providers import LLMProvider
from spade_llm.utils import load_env_vars

# Set up logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Load environment variables
    env_vars = load_env_vars()
    
    # Configuration
    XMPP_SERVER = "localhost"  # or your XMPP server
    AGENT_JID = f"agent@{XMPP_SERVER}"
    EXPERT_JID = f"expert@{XMPP_SERVER}"
    USER_JID = f"user@{XMPP_SERVER}"
    
    # Create OpenAI provider
    provider = LLMProvider.create_openai(
        api_key=env_vars["OPENAI_API_KEY"],
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # System prompt encouraging human consultation
    system_prompt = """You are an AI assistant with access to human experts.
    
    When you need:
    - Current information not in your training data
    - Human judgment or opinions
    - Company-specific information
    - Clarification on ambiguous requests
    
    Use the ask_human_expert tool to consult with human experts."""
    
    # Create human consultation tool
    human_tool = HumanInTheLoopTool(
        human_expert_jid=EXPERT_JID,
        timeout=300.0,  # 5 minutes
        name="ask_human_expert",
        description="Ask human expert for current info or clarification"
    )
    
    # Create LLM agent with human tool
    agent = LLMAgent(
        jid=AGENT_JID,
        password="agent_password",
        provider=provider,
        system_prompt=system_prompt,
        tools=[human_tool],
        verify_security=False
    )
    
    # Create chat interface for testing
    chat_agent = ChatAgent(
        jid=USER_JID,
        password="user_password",
        target_agent_jid=AGENT_JID,
        verify_security=False
    )
    
    # Start agents
    await agent.start()
    await chat_agent.start()
    
    print("\n" + "="*50)
    print("Human-in-the-Loop Example Running")
    print("="*50)
    print(f"Agent: {AGENT_JID}")
    print(f"Expert: {EXPERT_JID}")
    print(f"User: {USER_JID}")
    print("\n📋 Try these questions:")
    print("• 'What's the current weather in Madrid?'")
    print("• 'Should we proceed with the new project?'")
    print("• 'What's our company WiFi password?'")
    print("\n🌐 Make sure human expert is connected at:")
    print("   http://localhost:8080")
    print("\n💬 Type 'exit' to quit\n")
    
    # Run interactive chat
    try:
        await chat_agent.run_interactive(
            input_prompt="You: ",
            exit_command="exit",
            response_timeout=120.0
        )
    except KeyboardInterrupt:
        pass
    finally:
        await chat_agent.stop()
        await agent.stop()

if __name__ == "__main__":
    print("Starting Human-in-the-Loop example...")
    print("Make sure to start the human expert interface:")
    print("  python -m spade_llm.human_interface.web_server")
    print()
    spade.run(main())
```

**Key Features Demonstrated:**

- **🧠 Human Expert Tool**: Seamless integration with `HumanInTheLoopTool`
- **⚡ Real-time Communication**: XMPP messaging between agent and human
- **🌐 Web Interface**: Browser-based interface for human experts
- **🔄 Message Correlation**: Thread-based message routing
- **⏱️ Timeout Handling**: Graceful handling of delayed responses

**Setup Instructions:**

1. **Start web interface**: `python -m spade_llm.human_interface.web_server`
2. **Open browser**: Go to `http://localhost:8080`
3. **Connect as expert**: Use expert credentials to connect
4. **Run example**: Execute the Python script
5. **Test consultation**: Ask questions that require human input

Refer to the complete code example above or the [Human-in-the-Loop guide](../guides/human-in-the-loop.md) for detailed setup instructions.

## Basic Examples

### Simple Chat Agent

```python
import spade
from spade_llm import LLMAgent, ChatAgent, LLMProvider

async def main():
    # Create LLM provider
    provider = LLMProvider.create_openai(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    # Create LLM agent
    llm_agent = LLMAgent(
        jid="assistant@jabber.at",
        password="password1",
        provider=provider,
        system_prompt="You are a helpful assistant"
    )
    
    # Create chat interface
    chat_agent = ChatAgent(
        jid="human@jabber.at",
        password="password2",
        target_agent_jid="assistant@jabber.at"
    )
    
    # Start agents
    await llm_agent.start()
    await chat_agent.start()
    
    print("Type messages to chat. Enter 'exit' to quit.")
    await chat_agent.run_interactive()
    
    # Cleanup
    await chat_agent.stop()
    await llm_agent.stop()

if __name__ == "__main__":
    spade.run(main())
```

### Tool-Enabled Agent

```python
import spade
from spade_llm import LLMAgent, LLMProvider, LLMTool
from datetime import datetime
import requests

# Tool functions
async def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # Simplified weather API call
    try:
        response = requests.get(f"http://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={city}")
        data = response.json()
        return f"Weather in {city}: {data['current']['temp_c']}°C, {data['current']['condition']['text']}"
    except:
        return f"Could not get weather for {city}"

async def get_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def main():
    # Create tools
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
    
    time_tool = LLMTool(
        name="get_time",
        description="Get current date and time",
        parameters={"type": "object", "properties": {}, "required": []},
        func=get_time
    )
    
    # Create provider
    provider = LLMProvider.create_openai(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    # Create agent with tools
    agent = LLMAgent(
        jid="assistant@jabber.at",
        password="password",
        provider=provider,
        system_prompt="You are a helpful assistant with access to weather and time information",
        tools=[weather_tool, time_tool]
    )
    
    await agent.start()
    print("Agent with tools started!")
    
    # Keep running
    import asyncio
    await asyncio.sleep(60)
    
    await agent.stop()

if __name__ == "__main__":
    spade.run(main())
```

### Multi-Agent Workflow

```python
import spade
from spade_llm import LLMAgent, LLMProvider

# Routing functions
def analyzer_router(msg, response, context):
    """Route analysis results to reviewer."""
    if "analysis complete" in response.lower():
        return "reviewer@jabber.at"
    return str(msg.sender)

def reviewer_router(msg, response, context):
    """Route review results to executor."""
    if "approved" in response.lower():
        return "executor@jabber.at"
    elif "rejected" in response.lower():
        return "analyzer@jabber.at"  # Send back for revision
    return str(msg.sender)

async def main():
    provider = LLMProvider.create_openai(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    # Analyzer agent
    analyzer = LLMAgent(
        jid="analyzer@jabber.at",
        password="password1",
        provider=provider,
        system_prompt="You analyze requests and provide detailed analysis. End with 'Analysis complete.'",
        routing_function=analyzer_router
    )
    
    # Reviewer agent  
    reviewer = LLMAgent(
        jid="reviewer@jabber.at",
        password="password2",
        provider=provider,
        system_prompt="You review analysis and either approve or reject. Say 'Approved' or 'Rejected'.",
        routing_function=reviewer_router
    )
    
    # Executor agent
    executor = LLMAgent(
        jid="executor@jabber.at",
        password="password3",
        provider=provider,
        system_prompt="You execute approved plans and report completion."
    )
    
    # Start all agents
    await analyzer.start()
    await reviewer.start() 
    await executor.start()
    
    print("Multi-agent workflow started!")
    print("Send a request to analyzer@jabber.at to start the workflow")
    
    # Keep running
    import asyncio
    await asyncio.sleep(120)
    
    # Cleanup
    await analyzer.stop()
    await reviewer.stop()
    await executor.stop()

if __name__ == "__main__":
    spade.run(main())
```

### Local Model with Ollama

```python
import spade
from spade_llm import LLMAgent, LLMProvider

async def main():
    # Create Ollama provider
    provider = LLMProvider.create_ollama(
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
        temperature=0.7,
        timeout=120.0
    )
    
    # Create agent
    agent = LLMAgent(
        jid="local-agent@jabber.at",
        password="password",
        provider=provider,
        system_prompt="You are a helpful assistant running on a local model"
    )
    
    await agent.start()
    print("Local Ollama agent started!")
    
    # Keep running
    import asyncio
    await asyncio.sleep(60)
    
    await agent.stop()

if __name__ == "__main__":
    spade.run(main())
```

### Conversation Management

```python
import spade
from spade_llm import LLMAgent, LLMProvider

def conversation_ended(conversation_id: str, reason: str):
    """Handle conversation end."""
    print(f"Conversation {conversation_id} ended: {reason}")
    # Save conversation, send notifications, etc.

async def main():
    provider = LLMProvider.create_openai(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    agent = LLMAgent(
        jid="managed-agent@jabber.at",
        password="password",
        provider=provider,
        system_prompt="You are a helpful assistant. Say 'DONE' when tasks are complete.",
        max_interactions_per_conversation=5,  # Limit conversation length
        termination_markers=["DONE", "COMPLETE", "FINISHED"],
        on_conversation_end=conversation_ended
    )
    
    await agent.start()
    print("Agent with conversation management started!")
    
    # Test conversation state
    import asyncio
    await asyncio.sleep(30)
    
    # Check conversation states
    # In a real application, you'd have actual conversation IDs
    print("Active conversations:", len(agent.context.get_active_conversations()))
    
    await agent.stop()

if __name__ == "__main__":
    spade.run(main())
```


## Integration Examples

### With LangChain Tools

```python
from langchain_community.tools import DuckDuckGoSearchRun
from spade_llm.tools import LangChainToolAdapter

# Create LangChain tool
search_tool_lc = DuckDuckGoSearchRun()

# Adapt for SPADE_LLM
search_tool = LangChainToolAdapter(search_tool_lc)

# Use with agent
agent = LLMAgent(
    jid="search-agent@jabber.at",
    password="password",
    provider=provider,
    system_prompt="You are a research assistant with web search capabilities",
    tools=[search_tool]
)
```

## RAG System Examples

### Basic RAG Pipeline

Complete example demonstrating RAG components with Ollama and ChromaDB:

```python
"""
RAG System Example with Ollama and ChromaDB

Demonstrates document loading, splitting, embedding, and retrieval.

Prerequisites:
1. pip install spade_llm[chroma]
2. ollama pull nomic-embed-text
3. ollama serve
"""

import asyncio
from spade_llm.rag import (
    Document,
    Chroma,
    RecursiveCharacterTextSplitter,
    VectorStoreRetriever,
)
from spade_llm.providers import LLMProvider
from rich.console import Console

console = Console()

async def main():
    # 1. Create sample documents
    documents = [
        Document(
            content="Ollama enables running LLMs locally with privacy and control.",
            metadata={"topic": "ollama", "id": "1"}
        ),
        Document(
            content="Text embeddings are vector representations of semantic meaning.",
            metadata={"topic": "embeddings", "id": "2"}
        ),
        Document(
            content="SPADE is a multi-agent system platform based on XMPP.",
            metadata={"topic": "spade", "id": "3"}
        )
    ]
    
    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    console.print(f"[blue]Created {len(chunks)} chunks[/blue]")
    
    # 3. Setup embedding provider
    provider = LLMProvider.create_ollama(
        model="nomic-embed-text",
        base_url="http://localhost:11434/v1"
    )
    
    # 4. Initialize vector store
    vector_store = Chroma(
        collection_name="rag_example",
        embedding_fn=provider.get_embeddings
    )
    await vector_store.initialize()
    
    # 5. Index documents
    await vector_store.add_documents(chunks)
    count = await vector_store.get_document_count()
    console.print(f"[green]Indexed {count} documents[/green]")
    
    # 6. Create retriever and search
    retriever = VectorStoreRetriever(vector_store=vector_store)
    
    queries = [
        "How do embeddings work?",
        "What is SPADE?",
        "Running models locally"
    ]
    
    for query in queries:
        console.print(f"\n[yellow]Query:[/yellow] {query}")
        results = await retriever.retrieve(query, k=2)
        
        for i, doc in enumerate(results, 1):
            console.print(f"  {i}. {doc.content[:100]}...")
    
    # Cleanup
    await vector_store.cleanup()
    console.print("\n[green]Done![/green]")

if __name__ == "__main__":
    asyncio.run(main())
```

### RAG vs No RAG Comparison

See `examples/rag_vs_no_rag_demo.py` for a complete comparison demonstrating:

- Loading classic literature from Project Gutenberg
- Indexing documents with ChromaDB
- Comparing LLM responses with and without RAG
- Measuring accuracy improvements

**Key features demonstrated**:
- Document loading and preprocessing
- Vector store operations
- Semantic retrieval
- Results comparison

## Running Examples

1. **Install dependencies**: `pip install spade_llm`
2. **For RAG examples**: `pip install spade_llm[chroma]`, `ollama pull nomic-embed-text` 
3. **Set environment variables**: `export OPENAI_API_KEY="your-key"`
4. **Run example**: `python example.py`

## Common Patterns

### Environment Configuration

```python
import os
from spade_llm.utils import load_env_vars

# Load .env file
load_env_vars()

# Use environment variables
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```