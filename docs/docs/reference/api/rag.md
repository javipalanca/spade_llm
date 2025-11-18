# RAG API Reference

Complete API documentation for SPADE-LLM's RAG (Retrieval-Augmented Generation) components.

## Document Class

### `spade_llm.rag.Document`

Core data structure for representing documents in the RAG system.

```python
from spade_llm.rag import Document

doc = Document(
    content="Document text content",
    metadata={"source": "file.txt", "author": "user"}
)
```

#### Attributes

- **`content`** (str): The text content of the document
- **`metadata`** (dict, optional): Dictionary containing metadata about the document. Default: `{}`
- **`id`** (str, optional): Unique identifier for the document. Auto-generated UUID if not provided

#### Methods

- **`to_dict()`**: Convert the Document to a dictionary with keys "id", "content", "metadata"
- **`from_dict(data: dict)`** (classmethod): Create a Document instance from a dictionary

#### Properties

- **`text`**: Alias for the `content` attribute

#### Example

```python
doc = Document(
    content="SPADE-LLM enables building multi-agent LLM systems.",
    metadata={
        "source": "documentation.md",
        "section": "introduction",
        "timestamp": "2024-01-15",
        "version": "1.0"
    }
)

print(doc.content)  # Access content
print(doc.text)  # Same as content
print(doc.metadata["source"])
print(doc.id)

doc_dict = doc.to_dict()
new_doc = Document.from_dict(doc_dict)
```

---

## Document Loaders

### `spade_llm.rag.BaseDocumentLoader`

Abstract base class for document loaders.

```python
class BaseDocumentLoader:
    async def load_stream(self) -> AsyncGenerator[Document, None]:
        """Stream documents from the source as an async generator."""
        raise NotImplementedError
    
    async def load(self) -> list[Document]:
        """Load all documents from the source into a list."""
        return [doc async for doc in self.load_stream()]
```

### `spade_llm.rag.TextLoader`

Load documents from a single text file.

```python
from spade_llm.rag import TextLoader

loader = TextLoader(file_path="README.md", encoding="utf-8")
documents = await loader.load()
```

#### Parameters

- **`file_path`** (str): Path to the text file to load
- **`encoding`** (str, optional): File encoding. Default: `"utf-8"`

#### Returns

- **list[Document]**: List containing a single Document with file content

#### Example

```python
loader = TextLoader(file_path="./data/article.txt")
documents = await loader.load()

print(f"Loaded {len(documents)} document")
print(f"Content length: {len(documents[0].content)} characters")
print(f"Source: {documents[0].metadata['source']}")
```

### `spade_llm.rag.DirectoryLoader`

Load multiple documents from a directory.

```python
from spade_llm.rag import DirectoryLoader

loader = DirectoryLoader(
    path="./docs",
    glob_pattern="**/*.md"
)
documents = await loader.load()
```

#### Parameters

- **`path`** (str): Directory path to load documents from
- **`glob_pattern`** (str, optional): Glob pattern to match files. Default: `"**/*"`
- **`recursive`** (bool, optional): If True, search subdirectories. Overridden by glob_pattern. Default: `True`
- **`suffixes`** (list[str], optional): List of file suffixes to include (e.g., `['.txt', '.md']`). If None, uses all extensions from loader_map
- **`encoding`** (str, optional): File encoding. Default: `"utf-8"`
- **`metadata`** (dict, optional): Base metadata to attach to all loaded documents
- **`loader_map`** (dict, optional): A map of file extensions to loader classes

#### Returns

- **list[Document]**: List of Documents, one per matched file

#### Glob Patterns

```python
# All text files recursively
"**/*.txt"

# All markdown files in current directory only
"*.md"

# Multiple file types
"**/*.{md,txt,rst}"

# Specific subdirectory
"docs/**/*.md"
```

#### Example

```python
loader = DirectoryLoader(
    path="./documentation",
    glob_pattern="**/*.{md,rst,txt}"
)

documents = await loader.load()
print(f"Loaded {len(documents)} documents)"}

# Access metadata
for doc in documents[:3]:
    source = doc.metadata.get("source", "unknown")
    print(f"- {source}: {len(doc.content)} chars")
```

---

## Text Splitters

### `spade_llm.rag.TextSplitter`

Abstract base class for text splitters.

```python
class TextSplitter:
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        raise NotImplementedError
```

### `spade_llm.rag.CharacterTextSplitter`

Split documents by character count with optional separators.

```python
from spade_llm.rag import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n\n"
)
chunks = splitter.split_documents(documents)
```

#### Parameters

- **`chunk_size`** (int): Maximum size of each chunk in characters. Default: `2000`
- **`chunk_overlap`** (int): Number of overlapping characters between chunks. Default: `200`
- **`separator`** (str, optional): String to split on. Default: `"\n\n"`
- **`is_separator_regex`** (bool, optional): Whether separator is a regex pattern. Default: `False`
- **`keep_separator`** (bool | Literal["start", "end"], optional): Whether to keep the separator and where to place it. Can be False, True (equivalent to "start"), "start", or "end". Default: `False`
- **`length_function`** (callable, optional): Function to measure chunk length. Default: `len`
- **`add_start_index`** (bool, optional): If True, includes chunk's start index in metadata. Default: `True`
- **`strip_whitespace`** (bool, optional): If True, strips whitespace from chunks. Default: `True`

#### Returns

- **list[Document]**: List of Document chunks with preserved metadata

#### Example

```python
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n\n"
)

documents = [
    Document(
        content="Long document text...",
        metadata={"source": "doc.txt"}
    )
]

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Metadata is preserved in each chunk
for chunk in chunks:
    print(chunk.metadata["source"])  # Same as original
```

### `spade_llm.rag.RecursiveCharacterTextSplitter`

Intelligently split documents using a hierarchy of separators.

```python
from spade_llm.rag import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

#### Parameters

- **`chunk_size`** (int): Maximum size of each chunk in characters. Default: `2000`
- **`chunk_overlap`** (int): Number of overlapping characters between chunks. Default: `200`
- **`separators`** (list[str], optional): Ordered list of separators to try. Default: `["\n\n", "\n", " ", ""]`
- **`keep_separator`** (bool | Literal["start", "end"], optional): Whether and where to keep the separator. Default: `True`
- **`is_separator_regex`** (bool, optional): Whether separators are regex patterns. Default: `False`
- **`length_function`** (callable, optional): Function to measure chunk length. Default: `len`
- **`add_start_index`** (bool, optional): If True, includes chunk's start index in metadata. Default: `True`
- **`strip_whitespace`** (bool, optional): If True, strips whitespace from chunks. Default: `True`

#### Returns

- **list[Document]**: List of Document chunks

#### How It Works

1. Try splitting by first separator (e.g., `"\n\n"` for paragraphs)
2. If chunks are still too large, try next separator (e.g., `"\n"` for lines)
3. Continue through separator list until chunks are appropriate size
4. Preserves document structure better than character splitting

#### Example

```python
# Recommended for most use cases
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = [
    Document(content="""
# Introduction

This is a paragraph about RAG systems.
They combine retrieval with generation.

## Benefits

RAG provides several advantages:
- Better accuracy
- Up-to-date information
- Reduced hallucinations
    """)
]

chunks = splitter.split_documents(documents)

# Each chunk respects document structure
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.content[:100])
    print()
```

---

## Vector Stores

### `spade_llm.rag.VectorStore`

Abstract base class for vector stores.

```python
class VectorStore:
    async def initialize(self): ...
    async def add_documents(self, documents: list[Document]): ...
    async def similarity_search(self, query: str, k: int = 4): ...
    async def delete_collection(self): ...
    async def cleanup(self): ...
```

### `spade_llm.rag.Chroma`

ChromaDB vector store implementation.

```python
from spade_llm.rag import Chroma
from spade_llm.providers import LLMProvider

provider = LLMProvider.create_ollama(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="my_collection",
    embedding_fn=provider.get_embeddings,
    persist_directory="./vector_db"
)

await vector_store.initialize()
```

#### Parameters

- **`collection_name`** (str): Name of the ChromaDB collection. Default: `"documents"`
- **`persist_directory`** (str, optional): Directory to persist the database (for persistent client). If None, uses in-memory storage
- **`host`** (str, optional): Host for ChromaDB server (for HTTP client)
- **`port`** (int, optional): Port for ChromaDB server (for HTTP client). Default: `8000`
- **`ssl`** (bool, optional): Whether to use SSL for HTTP client. Default: `False`
- **`headers`** (dict, optional): Optional HTTP headers for HTTP client
- **`tenant`** (str, optional): Tenant ID. Default: `'default_tenant'`
- **`database`** (str, optional): Database name. Default: `'default_database'`
- **`embedding_fn`** (callable, optional): Async function that generates embeddings. Signature: `async def(texts: list[str]) -> list[list[float]]`
- **`collection_metadata`** (dict, optional): Optional metadata for the collection
- **`collection_configuration`** (CreateCollectionConfiguration, optional): Configuration for the collection. Use `chromadb.api.collection_configuration.CreateCollectionConfiguration` to define index properties (e.g., distance function). Example: `CreateCollectionConfiguration(hnsw={"space": "cosine"})`
- **`client_settings`** (Settings, optional): Optional Chroma client settings (`chromadb.config.Settings`)
- **`client`** (optional): Optional pre-configured Chroma client
- **`relevance_score_fn`** (callable, optional): Function to convert distance to relevance score. If None, auto-selected based on collection's distance function

#### Methods

##### `initialize()`

Initialize the vector store and collection.

```python
await vector_store.initialize()
```

**Returns**: None

##### `add_documents(documents: list[Document])`

Add documents to the vector store.

```python
documents = [
    Document(content="Text 1", metadata={"id": "doc1"}),
    Document(content="Text 2", metadata={"id": "doc2"})
]

await vector_store.add_documents(documents)
```

**Parameters**:

- `documents` (list[Document]): Documents to add

**Returns**: None

##### `similarity_search(query: str, k: int = 4, where: dict = None)`

Search for similar documents.

```python
results = await vector_store.similarity_search(
    query="What is RAG?",
    k=5,
    where={"category": "documentation"}
)
```

**Parameters**:

- `query` (str): Search query text
- `k` (int): Number of results to return. Default: 4
- `where` (dict, optional): Metadata filters

**Returns**: list[Document]

##### `similarity_search_with_score(query: str, k: int = 4, where: dict = None)`

Search with similarity scores.

```python
results = await vector_store.similarity_search_with_score(
    query="embeddings",
    k=3
)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {doc.content[:100]}")
```

**Parameters**:

- `query` (str): Search query text
- `k` (int): Number of results to return. Default: 4
- `where` (dict, optional): Metadata filters

**Returns**: list[tuple[Document, float]] - List of (document, score) tuples

##### `max_marginal_relevance_search(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filters: dict = None, **kwargs)`

Return documents selected using maximal marginal relevance.

```python
results = await vector_store.max_marginal_relevance_search(
    query="RAG systems",
    k=5,
    fetch_k=20,
    lambda_mult=0.5
)
```

**Parameters**:

- `query` (str): Search query text
- `k` (int): Number of documents to return. Default: 4
- `fetch_k` (int): Number of documents to fetch for MMR algorithm. Default: 20
- `lambda_mult` (float): Balance between diversity (0) and similarity (1). Default: 0.5
- `filters` (dict, optional): Filter by metadata
- `**kwargs`: Additional keyword arguments

**Returns**: list[Document]

##### `get_by_ids(ids: list[str])`

Get documents by their IDs.

```python
docs = await vector_store.get_by_ids(["id1", "id2", "id3"])
```

**Parameters**:

- `ids` (list[str]): List of document IDs to retrieve

**Returns**: list[Document]

##### `get(ids: list[str] = None, where: dict = None, limit: int = None, offset: int = None, where_document: dict = None, include: list[str] = None)`

Get documents by metadata.

```python
result = await vector_store.get(
    where={"source": "manual.pdf"},
    limit=10
)

ids = result.get("ids", [])
metadatas = result.get("metadatas", [])
documents = result.get("documents", [])
```

**Parameters**:

- `ids` (list[str], optional): Document IDs to retrieve
- `where` (dict, optional): Filter results by metadata. E.g. `{"color": "red", "price": {"$lt": 5.0}}`
- `limit` (int, optional): Number of documents to return
- `offset` (int, optional): Offset to start returning results from. Useful for paging results with limit
- `where_document` (dict, optional): Filter by document content. E.g. `{"$contains": "hello"}`
- `include` (list[str], optional): What to include in the results. Can contain "embeddings", "metadatas", "documents". IDs are always included. Defaults to `["metadatas", "documents"]`

**Returns**: dict with keys "ids", "embeddings" (if requested), "metadatas", "documents"

##### `update_document(document_id: str, document: Document)`

Update a single document in the collection.

```python
updated_doc = Document(
    content="Updated content",
    metadata={"version": "2.0"}
)
await vector_store.update_document("doc_id", updated_doc)
```

**Parameters**:

- `document_id` (str): ID of the document to update
- `document` (Document): New document content

**Returns**: None

##### `update_documents(ids: list[str], documents: list[Document])`

Update multiple documents in the collection.

```python
await vector_store.update_documents(
    ["id1", "id2"],
    [doc1, doc2]
)
```

**Parameters**:

- `ids` (list[str]): List of document IDs to update
- `documents` (list[Document]): List of new document contents

**Returns**: None

##### `delete(ids: list[str])`

Delete documents by ID.

```python
# Get IDs to delete
result = await vector_store.get(where={"version": "old"})
ids = result.get("ids", [])

# Delete them
success = await vector_store.delete(ids)
print(f"Deleted: {success}")
```

**Parameters**:

- `ids` (list[str]): Document IDs to delete

**Returns**: bool - True if successful

##### `delete_collection()`

Delete the entire collection.

```python
await vector_store.delete_collection()
```

**Returns**: None

##### `reset_collection()`

Reset the collection by deleting and recreating it. Useful for testing or when you want to completely clear and reinitialize a collection.

```python
await vector_store.reset_collection()
```

**Returns**: None

##### `from_documents(documents: list[Document], embedding_fn: callable = None, collection_name: str = "documents", **kwargs)` (classmethod)

Create a Chroma vector store from a list of documents.

```python
store = await Chroma.from_documents(
    documents=docs,
    embedding_fn=provider.get_embeddings,
    collection_name="my_docs",
    persist_directory="./db"
)
```

**Parameters**:

- `documents` (list[Document]): List of Document objects to add
- `embedding_fn` (callable, optional): Async function that takes a list of texts and returns embeddings
- `collection_name` (str): Name of the collection to create. Default: `"documents"`
- `**kwargs`: Additional arguments to pass to Chroma constructor

**Returns**: Initialized Chroma vector store with documents added

##### `from_texts(texts: list[str], embedding_fn: callable = None, metadatas: list[dict] = None, ids: list[str] = None, collection_name: str = "documents", **kwargs)` (classmethod)

Create a Chroma vector store from a list of texts.

```python
store = await Chroma.from_texts(
    texts=["Text 1", "Text 2"],
    embedding_fn=provider.get_embeddings,
    metadatas=[{"source": "a"}, {"source": "b"}]
)
```

**Parameters**:

- `texts` (list[str]): List of text strings to add
- `embedding_fn` (callable, optional): Async function that takes a list of texts and returns embeddings
- `metadatas` (list[dict], optional): Optional list of metadata dicts for each text
- `ids` (list[str], optional): Optional list of IDs for each text
- `collection_name` (str): Name of the collection to create. Default: `"documents"`
- `**kwargs`: Additional arguments to pass to Chroma constructor

**Returns**: Initialized Chroma vector store with texts added

##### `get_document_count()`

Get total number of documents in collection.

```python
count = await vector_store.get_document_count()
print(f"Total documents: {count}")
```

**Returns**: int

##### `cleanup()`

Clean up resources.

```python
await vector_store.cleanup()
```

**Returns**: None

#### Complete Example

```python
from spade_llm.rag import Chroma, Document
from spade_llm.providers import LLMProvider

async def main():
    # Setup
    provider = LLMProvider.create_ollama(model="nomic-embed-text")
    
    vector_store = Chroma(
        collection_name="docs",
        embedding_fn=provider.get_embeddings,
        persist_directory="./my_db"
    )
    
    await vector_store.initialize()
    
    # Add documents
    doc1 = Document(
        content="RAG combines retrieval and generation",
        metadata={"topic": "rag"}
    )
    doc2 = Document(
        content="Vector stores enable semantic search",
        metadata={"topic": "vectors"}
    )
    docs = [doc1, doc2]
    
    await vector_store.add_documents(docs)
    
    # Search
    results = await vector_store.similarity_search("What is RAG?", k=2)
    for doc in results:
        print(doc.content)
    
    # Search with scores
    scored_results = await vector_store.similarity_search_with_score(
        "semantic search", k=1
    )
    for doc, score in scored_results:
        print(f"Score: {score:.3f} - {doc.content}")
    
    # Get by metadata
    rag_docs = await vector_store.get(where={"topic": "rag"})
    print(f"Found {len(rag_docs['ids'])} RAG documents")
    
    # Delete specific documents by ID
    await vector_store.delete([doc1.id])
    
    # Check count
    count = await vector_store.get_document_count()
    print(f"Remaining: {count} documents")
    
    # Cleanup
    await vector_store.cleanup()
```

---

## Retrievers

### `spade_llm.rag.BaseRetriever`

Abstract base class for retrievers.

```python
class BaseRetriever:
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> list[Document]:
        """Retrieve relevant documents for query."""
        raise NotImplementedError
```

### `spade_llm.rag.VectorStoreRetriever`

Retriever that uses a vector store for document retrieval.

```python
from spade_llm.rag import VectorStoreRetriever

retriever = VectorStoreRetriever(
    vector_store=vector_store
)

results = await retriever.retrieve(
    query="How do I configure agents?",
    k=5,
    search_type="similarity"
)
```

#### Parameters

- **`vector_store`** (VectorStore): Initialized vector store instance

#### Methods

##### `retrieve_similarity(query: str, k: int = 4, sim_threshold: float = float("inf"), filters: dict = None, **kwargs)`

Retrieve documents based on vector similarity.

```python
results = await retriever.retrieve_similarity(
    query="agent communication patterns",
    k=5,
    sim_threshold=0.7,
    filters={"category": "architecture"}
)
```

**Parameters**:

- `query` (str): Search query text
- `k` (int): Number of documents to retrieve. Default: 4
- `sim_threshold` (float): Minimum similarity score threshold. Documents with scores below this will be filtered out. Use `float("inf")` to disable filtering. Default: `float("inf")`
- `filters` (dict, optional): Optional metadata filters to narrow down the search
- `**kwargs`: Additional keyword arguments passed to the vector store

**Returns**: list[Document]

##### `retrieve_mmr(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filters: dict = None, **kwargs)`

Retrieve documents using Maximal Marginal Relevance for diversity.

```python
results = await retriever.retrieve_mmr(
    query="LLM providers",
    k=5,
    fetch_k=20,
    lambda_mult=0.5
)
```

**Parameters**:

- `query` (str): Search query text
- `k` (int): Number of documents to return. Default: 4
- `fetch_k` (int): Number of documents to fetch for MMR algorithm. The algorithm will first fetch this many documents, then select k diverse documents from them. Default: 20
- `lambda_mult` (float): Balance between similarity (1.0) and diversity (0.0). Default: 0.5
- `filters` (dict, optional): Optional metadata filters to narrow down the search
- `**kwargs`: Additional keyword arguments passed to the vector store

**Returns**: list[Document]

**Raises**: NotImplementedError if the underlying vector store doesn't support MMR

##### `retrieve(query: str, k: int = 4, search_type: str = "similarity", **kwargs)`

Generic retrieval method that dispatches to specific search methods.

```python
results = await retriever.retrieve(
    query="What is multi-agent coordination?",
    k=10,
    search_type="similarity",
    filters={"category": "architecture"}
)
```

**Parameters**:

- `query` (str): Search query text
- `k` (int): Number of documents to retrieve. Default: 4
- `search_type` (str): Type of search to perform. Options: "similarity" (default) or "mmr"
- `**kwargs`: Additional arguments passed to the specific retrieval method. For similarity: `sim_threshold`, `filters`. For mmr: `fetch_k`, `lambda_mult`, `filters`

**Returns**: list[Document]

**Note**: For a more explicit and type-safe API, prefer using the specific methods: `retrieve_similarity()` or `retrieve_mmr()`

**Search Types**:

- **`"similarity"`**: Standard similarity search
- **`"mmr"`**: Maximal Marginal Relevance (balances relevance and diversity)

#### Examples

**Basic Similarity Search**:

```python
results = await retriever.retrieve(
    query="agent communication patterns",
    k=5
)

for doc in results:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.content[:200]}...")
    print()
```

**MMR Search (Diversity)**:

```python
# Get diverse results
results = await retriever.retrieve(
    query="LLM providers",
    k=5,
    search_type="mmr",
    fetch_k=20,  # Fetch more candidates
    lambda_mult=0.5  # Balance relevance (1.0) vs diversity (0.0)
)
```

**Filtered Retrieval**:

```python
# Only retrieve from specific sources
results = await retriever.retrieve(
    query="installation steps",
    k=10,
    filters={
        "section": "getting-started",
        "version": "latest"
    }
)
```

**Complete Workflow**:

```python
from spade_llm.rag import (
    Chroma,
    VectorStoreRetriever,
    RecursiveCharacterTextSplitter,
    DirectoryLoader
)
from spade_llm.providers import LLMProvider

async def setup_retrieval():
    # Load and chunk documents
    loader = DirectoryLoader(path="./docs", glob_pattern="**/*.md")
    documents = await loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    
    # Initialize vector store
    provider = LLMProvider.create_ollama(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="documentation",
        embedding_fn=provider.get_embeddings
    )
    await vector_store.initialize()
    await vector_store.add_documents(chunks)
    
    # Create retriever
    retriever = VectorStoreRetriever(vector_store=vector_store)
    
    # Retrieve documents
    results = await retriever.retrieve(
        query="How do I create a custom tool?",
        k=5,
        search_type="similarity"
    )
    
    return results
```

---

## See Also

- **[RAG System Guide](../../guides/rag-system.md)** - Comprehensive guide
- **[Examples](../examples.md)** - Working code examples
- **[Providers API](providers.md)** - Embedding generation
