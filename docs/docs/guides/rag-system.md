# Building RAG Systems with SPADE-LLM

This guide shows you how to build Retrieval-Augmented Generation (RAG) systems that enhance your LLM agents with external knowledge. You'll learn when to use RAG, how to design effective retrieval pipelines, and how to integrate them into multi-agent workflows.

## When Should You Use RAG?

Before diving into implementation, consider whether RAG is the right solution for your use case:

### ✅ Use RAG When:

- Your agents need **current or frequently updated information** that wasn't in the LLM's training data
- You're building **domain-specific assistants** (e.g., company documentation, legal docs, technical manuals)
- You need **verifiable, source-based answers** rather than model-generated responses
- Your knowledge base is **too large** to fit in the context window
- You want to **reduce hallucinations** by grounding responses in actual documents

### ❌ Skip RAG When:

- Your task requires only **general knowledge** already in the LLM
- You need **creative generation** without factual constraints
- Your use case is **simple Q&A** that doesn't require external context
- **Real-time performance** is critical and retrieval latency is unacceptable

## Understanding the RAG Pipeline

Building a RAG system involves three phases that work together:

### Phase 1: Indexing (Build Your Knowledge Base)

This happens once (or periodically) to prepare your documents:

1. **Load** documents from files, databases, or APIs
2. **Chunk** text into semantic pieces
3. **Embed** chunks into vectors using an embedding model
4. **Store** vectors in a database for fast similarity search

Think of this as creating a searchable library catalog for your documents.

### Phase 2: Retrieval (Find Relevant Information)

This happens at query time:

1. User asks a question
2. Convert question to a vector (using the same embedding model)
3. Search vector database for similar document chunks
4. Return the most relevant chunks (typically top 3-5)

This is like asking a librarian to find books on a specific topic.

### Phase 3: Generation (Create the Answer)

The LLM uses retrieved context:

1. Retrieved chunks are added to the LLM's prompt as context
2. LLM generates an answer based on this context
3. Response includes information from actual documents, not just model memory

This combines the librarian's retrieved books with the LLM's reading comprehension.

## Building Your First RAG System

Let's build a practical RAG system step by step. We'll create a documentation assistant that answers questions about your codebase.

### Step 1: Load Your Documents

Choose the loader that matches your document source:

```python
from spade_llm.rag import DirectoryLoader

# Load all markdown files from documentation
loader = DirectoryLoader(
    path="./docs",
    glob_pattern="**/*.md"
)
documents = await loader.load()
print(f"Loaded {len(documents)} documents")
```

**Tip**: Use metadata to organize documents by category, version, or author. This helps with filtering during retrieval.

### Step 2: Chunk Your Documents

Chunking is critical, too small and you lose context, too large and retrieval becomes imprecise.

```python
from spade_llm.rag import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

**Why overlap?** Overlap prevents important information from being split across chunk boundaries.

### Step 3: Choose Your Embedding Model

Embeddings convert text into vectors that capture semantic meaning. Two main options:

#### Option A: OpenAI Embeddings (Cloud)

```python
from spade_llm.providers import LLMProvider

embedding_provider = LLMProvider.create_openai(
    api_key="your-api-key",
    model="text-embedding-3-small"
)
```

**When to use**: You need highest quality, have budget, don't mind cloud dependency.

#### Option B: Ollama Embeddings (Local)

```python
# First: ollama pull nomic-embed-text
embedding_provider = LLMProvider.create_ollama(
    model="nomic-embed-text"
)
```

**When to use**: Privacy matters, offline operation needed, high-volume use case.

**Critical**: Always use the **same embedding model** for indexing and querying. Mixing models breaks semantic similarity.

### Step 4: Create and Populate Vector Store

```python
from spade_llm.rag import Chroma

# Initialize vector store with persistent storage
vector_store = Chroma(
    collection_name="docs_kb",
    embedding_fn=embedding_provider.get_embeddings,
    persist_directory="./vector_db"
)
await vector_store.initialize()

# Index all chunks (this may take a few minutes)
await vector_store.add_documents(chunks)
print(f"Indexed {await vector_store.get_document_count()} chunks")
```

### Step 5: Set Up Retrieval

Create a retriever to query your indexed documents:

```python
from spade_llm.rag import VectorStoreRetriever

retriever = VectorStoreRetriever(
    vector_store=vector_store,
    search_type="similarity",  # or "mmr" for diverse results
    k=5  # Return top 5 chunks
)

results = await retriever.retrieve("How do I configure agents?")
for doc in results:
    print(f"📄 {doc.metadata['source']}: {doc.content[:100]}...")
```

## Integrating RAG with your agents

### Pattern 1: Direct Agent Integration

For simple use cases, integrate retrieval directly into an agent workflow (this works with traditional SPADE agents):

```python
from spade_llm import RetrievalAgent
from spade_llm.rag import VectorStoreRetriever

# Create a retrieval agent that handles document queries
retrieval_agent = RetrievalAgent(
    jid="retrieval@localhost",
    password="password",
    retriever=VectorStoreRetriever(vector_store=vector_store)
)
await retrieval_agent.start()
```

**When to use**: Single knowledge base, simple architecture, getting started.

### Pattern 2: Multi-Agent RAG (Recommended)

In this case, LLM agents can query retrieval agents via tools:

```python
from spade_llm import LLMAgent, RetrievalAgent
from spade_llm.tools import RetrievalTool
from spade_llm.providers import LLMProvider

# 1. Start retrieval agent (manages the knowledge base)
retrieval_agent = RetrievalAgent(
    jid="retrieval@localhost",
    password="retrieval_pass",
    retriever=retriever
)
await retrieval_agent.start()

# 2. Create LLM agent with retrieval tool
llm_provider = LLMProvider.create_openai(api_key="your-key")
retrieval_tool = RetrievalTool(
    name="docs_search",
    description="Search technical documentation for code examples and explanations",
    retrieval_agent_jid="retrieval@localhost",
    k=5
)

llm_agent = LLMAgent(
    jid="assistant@localhost",
    password="assistant_pass",
    provider=llm_provider,
    tools=[retrieval_tool]
)
await llm_agent.start()

# 3. Query the agent - it will automatically use retrieval when needed
response = await llm_agent.query("How do I create custom behaviours?")
```

**How it works**:

1. User asks the LLM agent a question
2. LLM **decides** whether to search the knowledge base
3. If needed, it calls the `docs_search` tool
4. Tool sends XMPP message to `RetrievalAgent`
5. Retrieved documents are returned and used as context
6. LLM generates grounded response with sources

**When to use**: Production systems, multiple knowledge bases, need agent autonomy.

### Pattern 3: Distributed Knowledge Bases

Deploy multiple specialized retrieval agents for different domains:

```python
# Technical documentation retrieval agent
tech_retrieval = RetrievalAgent(
    jid="tech_docs@localhost",
    password="pass",
    retriever=tech_retriever
)

# HR policies retrieval agent  
hr_retrieval = RetrievalAgent(
    jid="hr_docs@localhost",
    password="pass",
    retriever=hr_retriever
)

# LLM agent with access to both
llm_agent = LLMAgent(
    jid="assistant@localhost",
    password="pass",
    provider=llm_provider,
    tools=[
        RetrievalTool(
            name="tech_docs",
            description="Search technical and API documentation",
            retrieval_agent_jid="tech_docs@localhost"
        ),
        RetrievalTool(
            name="hr_policies",
            description="Search HR policies and employee handbook",
            retrieval_agent_jid="hr_docs@localhost"
        )
    ]
)
```

**When to use**: Access control needs, domain separation.

## Next Steps

- **[RAG API Reference](../reference/api/rag.md)** - Detailed API documentation
- **[Examples](../reference/examples.md)** - Complete working examples
- **[Tools System](tools-system.md)** - Integrate RAG with LLM agents
- **[Providers](providers.md)** - Embedding model configuration
