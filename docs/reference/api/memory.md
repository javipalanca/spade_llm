# Memory API

API reference for SPADE_LLM's dual memory system supporting both interaction memory and agent base memory.

## AgentInteractionMemory

Manages conversation-specific memory with JSON-based storage.

### Constructor

```python
AgentInteractionMemory(
    agent_id: str,
    memory_path: Optional[str] = None
)
```

**Parameters:**
- `agent_id` (str): Unique identifier for the agent
- `memory_path` (str, optional): Custom memory storage directory path

**Example:**

```python
from spade_llm.memory import AgentInteractionMemory

# Default storage location
memory = AgentInteractionMemory("agent@example.com")

# Custom storage location
memory = AgentInteractionMemory(
    agent_id="agent@example.com",
    memory_path="/custom/memory/path"
)
```

### Methods

#### add_information()

```python
def add_information(
    self, 
    conversation_id: str, 
    information: str
) -> str
```

Add information to conversation memory.

**Parameters:**
- `conversation_id` (str): Unique conversation identifier
- `information` (str): Information to store

**Returns:**
- `str`: Confirmation message

**Example:**

```python
memory.add_information(
    "user1_session",
    "User prefers JSON responses over XML"
)
```

#### get_information()

```python
def get_information(
    self, 
    conversation_id: str
) -> List[str]
```

Get stored information for a conversation.

**Parameters:**
- `conversation_id` (str): Conversation identifier

**Returns:**
- `List[str]`: List of stored information strings

**Example:**

```python
info_list = memory.get_information("user1_session")
# Returns: ["User prefers JSON responses", "API token: abc123"]
```

#### get_context_summary()

```python
def get_context_summary(
    self, 
    conversation_id: str
) -> Optional[str]
```

Get formatted summary of conversation memory.

**Parameters:**
- `conversation_id` (str): Conversation identifier

**Returns:**
- `Optional[str]`: Formatted summary string or None if no memory

**Example:**

```python
summary = memory.get_context_summary("user1_session")
# Returns: "Previous interactions:\n- User prefers JSON responses\n- API token: abc123"
```

#### clear_conversation()

```python
def clear_conversation(
    self, 
    conversation_id: str
) -> bool
```

Clear memory for a specific conversation.

**Parameters:**
- `conversation_id` (str): Conversation identifier

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**

```python
success = memory.clear_conversation("user1_session")
```

#### get_all_conversations()

```python
def get_all_conversations(self) -> List[str]
```

Get list of all conversation IDs with stored memory.

**Returns:**
- `List[str]`: List of conversation IDs

**Example:**

```python
conversations = memory.get_all_conversations()
# Returns: ["user1_session", "user2_session", "api_session"]
```

## AgentBaseMemory

Manages long-term agent memory with SQLite backend and categorized storage. Supports both persistent file-based and temporary in-memory storage modes.

### Constructor

```python
AgentBaseMemory(
    agent_id: str,
    memory_path: Optional[str] = None,
    backend: Optional[MemoryBackend] = None
)
```

**Parameters:**
- `agent_id` (str): Unique identifier for the agent
- `memory_path` (str, optional): Custom memory storage directory path or ":memory:" for in-memory storage
- `backend` (MemoryBackend, optional): Custom memory backend implementation

**Example:**

```python
from spade_llm.memory import AgentBaseMemory

# Default SQLite backend (persistent)
memory = AgentBaseMemory("agent@example.com")

# Custom memory path (persistent)
memory = AgentBaseMemory(
    agent_id="agent@example.com",
    memory_path="/custom/memory/path"
)

# In-memory storage (temporary)
memory = AgentBaseMemory(
    agent_id="agent@example.com",
    memory_path=":memory:"
)
```

### Async Methods

#### initialize()

```python
async def initialize(self) -> None
```

Initialize the memory backend and create necessary database structures.

**Example:**

```python
await memory.initialize()
```

#### store_memory()

```python
async def store_memory(
    self,
    content: str,
    category: str,
    context: Optional[str] = None,
    confidence: float = 1.0
) -> str
```

Store information in long-term memory.

**Parameters:**
- `content` (str): Information content to store
- `category` (str): Memory category (`fact`, `pattern`, `preference`, `capability`)
- `context` (str, optional): Additional context information
- `confidence` (float, optional): Confidence score (0.0 to 1.0), defaults to 1.0

**Returns:**
- `str`: Unique memory identifier

**Example:**

```python
memory_id = await memory.store_memory(
    content="API endpoint requires authentication header",
    category="fact",
    context="API integration discussion",
    confidence=0.9
)
```

#### search_memories()

```python
async def search_memories(
    self,
    query: str,
    limit: int = 10,
    category: Optional[str] = None
) -> List[MemoryEntry]
```

Search through stored memories.

**Parameters:**
- `query` (str): Search query string
- `limit` (int, optional): Maximum number of results, defaults to 10
- `category` (str, optional): Filter by memory category

**Returns:**
- `List[MemoryEntry]`: List of matching memory entries

**Example:**

```python
results = await memory.search_memories(
    query="API authentication",
    limit=5,
    category="fact"
)
```

#### get_memories_by_category()

```python
async def get_memories_by_category(
    self,
    category: str,
    limit: int = 10
) -> List[MemoryEntry]
```

Get memories by category.

**Parameters:**
- `category` (str): Memory category to retrieve
- `limit` (int, optional): Maximum number of results, defaults to 10

**Returns:**
- `List[MemoryEntry]`: List of memory entries in category

**Example:**

```python
facts = await memory.get_memories_by_category("fact", limit=20)
```

#### get_recent_memories()

```python
async def get_recent_memories(
    self,
    limit: int = 10
) -> List[MemoryEntry]
```

Get most recently accessed memories.

**Parameters:**
- `limit` (int, optional): Maximum number of results, defaults to 10

**Returns:**
- `List[MemoryEntry]`: List of recent memory entries

**Example:**

```python
recent = await memory.get_recent_memories(limit=5)
```

#### update_memory()

```python
async def update_memory(
    self,
    memory_id: str,
    content: Optional[str] = None,
    category: Optional[str] = None,
    context: Optional[str] = None,
    confidence: Optional[float] = None
) -> bool
```

Update existing memory entry.

**Parameters:**
- `memory_id` (str): Memory identifier to update
- `content` (str, optional): New content
- `category` (str, optional): New category
- `context` (str, optional): New context
- `confidence` (float, optional): New confidence score

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**

```python
success = await memory.update_memory(
    memory_id="mem_123",
    content="Updated API endpoint information",
    confidence=0.95
)
```

#### delete_memory()

```python
async def delete_memory(
    self,
    memory_id: str
) -> bool
```

Delete a memory entry.

**Parameters:**
- `memory_id` (str): Memory identifier to delete

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**

```python
success = await memory.delete_memory("mem_123")
```

#### get_memory_stats()

```python
async def get_memory_stats(self) -> Dict[str, Any]
```

Get memory usage statistics.

**Returns:**
- `Dict[str, Any]`: Statistics dictionary

**Example:**

```python
stats = await memory.get_memory_stats()
# Returns: {
#     "total_memories": 150,
#     "categories": {"fact": 50, "pattern": 30, "preference": 40, "capability": 30},
#     "avg_confidence": 0.85,
#     "oldest_memory": "2025-01-01T00:00:00Z",
#     "newest_memory": "2025-01-10T12:00:00Z"
# }
```

#### cleanup()

```python
async def cleanup(self) -> None
```

Clean up memory backend resources.

**Example:**

```python
await memory.cleanup()
```

## MemoryEntry

Data structure representing a memory entry.

### Structure

```python
@dataclass
class MemoryEntry:
    id: str
    agent_id: str
    category: str
    content: str
    context: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
```

**Fields:**
- `id` (str): Unique memory identifier
- `agent_id` (str): Agent that owns this memory
- `category` (str): Memory category
- `content` (str): Memory content
- `context` (str, optional): Additional context
- `confidence` (float): Confidence score (0.0 to 1.0)
- `created_at` (datetime): Creation timestamp
- `last_accessed` (datetime): Last access timestamp
- `access_count` (int): Number of times accessed

### Example

```python
memory_entry = MemoryEntry(
    id="mem_123",
    agent_id="agent@example.com",
    category="fact",
    content="API requires authentication header",
    context="API integration discussion",
    confidence=0.9,
    created_at=datetime.now(),
    last_accessed=datetime.now(),
    access_count=3
)
```

## Memory Tools

### Interaction Memory Tools

#### remember_interaction_info

Tool function for storing conversation-specific information.

```python
def remember_interaction_info(information: str) -> str
```

**Parameters:**
- `information` (str): Information to store in memory

**Returns:**
- `str`: Confirmation message

**LLM Tool Schema:**

```json
{
    "name": "remember_interaction_info",
    "description": "Store important information for future reference in this conversation",
    "parameters": {
        "type": "object",
        "properties": {
            "information": {
                "type": "string",
                "description": "Important information to remember for this conversation"
            }
        },
        "required": ["information"]
    }
}
```

### Agent Base Memory Tools

#### store_memory

Tool function for storing information in long-term memory.

```python
def store_memory(
    content: str,
    category: str,
    context: Optional[str] = None
) -> str
```

**Parameters:**
- `content` (str): Information content to store
- `category` (str): Memory category (`fact`, `pattern`, `preference`, `capability`)
- `context` (str, optional): Additional context information

**Returns:**
- `str`: Confirmation message with memory ID

**LLM Tool Schema:**

```json
{
    "name": "store_memory",
    "description": "Store information in long-term memory with category classification",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Information content to store"
            },
            "category": {
                "type": "string",
                "enum": ["fact", "pattern", "preference", "capability"],
                "description": "Memory category for organization"
            },
            "context": {
                "type": "string",
                "description": "Additional context information (optional)"
            }
        },
        "required": ["content", "category"]
    }
}
```

#### search_memories

Tool function for searching through stored memories.

```python
def search_memories(
    query: str,
    limit: Optional[int] = 10,
    category: Optional[str] = None
) -> str
```

**Parameters:**
- `query` (str): Search query string
- `limit` (int, optional): Maximum number of results, defaults to 10
- `category` (str, optional): Filter by memory category

**Returns:**
- `str`: Formatted search results

**LLM Tool Schema:**

```json
{
    "name": "search_memories",
    "description": "Search through stored memories for relevant information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10)"
            },
            "category": {
                "type": "string",
                "enum": ["fact", "pattern", "preference", "capability"],
                "description": "Filter by memory category (optional)"
            }
        },
        "required": ["query"]
    }
}
```

#### list_memories

Tool function for listing memories by category.

```python
def list_memories(
    category: Optional[str] = None,
    limit: Optional[int] = 10
) -> str
```

**Parameters:**
- `category` (str, optional): Memory category to list, if None lists recent memories
- `limit` (int, optional): Maximum number of results, defaults to 10

**Returns:**
- `str`: Formatted list of memories

**LLM Tool Schema:**

```json
{
    "name": "list_memories",
    "description": "List memories by category or view recent memories",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["fact", "pattern", "preference", "capability"],
                "description": "Memory category to list (optional, lists recent if not specified)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10)"
            }
        },
        "required": []
    }
}
```

## Memory Backends

### MemoryBackend (Abstract Base Class)

Abstract interface for memory storage backends.

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class MemoryBackend(ABC):
    """Abstract base class for memory storage backends"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend"""
        pass
    
    @abstractmethod
    async def store_memory(self, entry: MemoryEntry) -> str:
        """Store a memory entry"""
        pass
    
    @abstractmethod
    async def search_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Search memories"""
        pass
    
    @abstractmethod
    async def get_memories_by_category(
        self,
        agent_id: str,
        category: str,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Get memories by category"""
        pass
    
    @abstractmethod
    async def get_recent_memories(
        self,
        agent_id: str,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Get recent memories"""
        pass
    
    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        **updates
    ) -> bool:
        """Update a memory entry"""
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        pass
    
    @abstractmethod
    async def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up backend resources"""
        pass
```

### SqliteBackend

SQLite implementation of the memory backend.

```python
from spade_llm.memory.backends.sqlite_backend import SqliteBackend

backend = SqliteBackend(database_path="/path/to/memory.db")
await backend.initialize()
```

**Features:**
- Async SQLite operations using `aiosqlite`
- Full-text search capabilities
- Optimized indexing for fast queries
- ACID compliance for data integrity
- Concurrent access support

## Storage Formats

### Interaction Memory Storage

**File Path**: `{memory_path}/{safe_agent_id}_interactions.json`

**JSON Structure:**
```json
{
    "agent_id": "agent@example.com",
    "interactions": {
        "conversation_id": [
            {
                "content": "User prefers JSON responses",
                "timestamp": "2025-01-10T10:30:00.000Z"
            },
            {
                "content": "API token: db_token_123",
                "timestamp": "2025-01-10T10:35:00.000Z"
            }
        ]
    }
}
```

### Agent Base Memory Storage

**File Path**: `{memory_path}/{safe_agent_id}_base_memory.db`

**SQLite Schema:**
```sql
CREATE TABLE agent_memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    context TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX idx_agent_category ON agent_memories(agent_id, category);
CREATE INDEX idx_content_search ON agent_memories(content);
CREATE INDEX idx_created_at ON agent_memories(created_at);
```

## LLMAgent Integration

### Configuration Parameters

```python
from spade_llm.agent import LLMAgent

agent = LLMAgent(
    jid="agent@example.com",
    password="password",
    provider=provider,
    
    # Memory configuration
    interaction_memory: Union[bool, Tuple[bool, str]] = False,
    agent_base_memory: Union[bool, Tuple[bool, str]] = False,
    
    # Memory path (alternative to tuple syntax)
    memory_path: Optional[str] = None,
    
    # System prompt with memory instructions
    system_prompt: str = "You have memory capabilities..."
)
```

### Memory Path Configuration

```python
# Boolean flags (uses default or environment path)
agent = LLMAgent(
    jid="agent@example.com",
    password="password",
    provider=provider,
    interaction_memory=True,
    agent_base_memory=True
)

# Tuple syntax with custom paths
agent = LLMAgent(
    jid="agent@example.com",
    password="password",
    provider=provider,
    interaction_memory=(True, "/custom/interaction/path"),
    agent_base_memory=(True, "/custom/base/path")
)

# Environment variable
import os
os.environ['SPADE_LLM_MEMORY_PATH'] = "/custom/memory/path"
```

### Tool Auto-Registration

When memory is enabled, tools are automatically registered:

```python
# Interaction memory tools
if interaction_memory:
    agent.register_tool(remember_interaction_info)

# Agent base memory tools
if agent_base_memory:
    agent.register_tool(store_memory)
    agent.register_tool(search_memories)
    agent.register_tool(list_memories)
```

## Example Usage

### Complete Integration Example

```python
import asyncio
from spade_llm.agent import LLMAgent
from spade_llm.providers import LLMProvider
from spade_llm.memory import AgentBaseMemory, AgentInteractionMemory

async def memory_integration_example():
    # Create provider
    provider = LLMProvider(
        model="gpt-5-nano",
        api_key="your-api-key",
    )
    
    # Create agent with both memory types
    agent = LLMAgent(
        jid="memory_agent@example.com",
        password="password",
        provider=provider,
        interaction_memory=True,
        agent_base_memory=True,
        system_prompt="""You are an assistant with dual memory capabilities.
        
        Use remember_interaction_info for conversation-specific information.
        Use store_memory, search_memories, and list_memories for long-term learning.
        """
    )
    
    # Access memory instances directly
    interaction_memory = agent.interaction_memory
    base_memory = agent.agent_base_memory
    
    # Direct memory operations
    if interaction_memory:
        interaction_memory.add_information("conv_1", "User prefers JSON")
        summary = interaction_memory.get_context_summary("conv_1")
        print(f"Interaction memory: {summary}")
    
    if base_memory:
        await base_memory.initialize()
        
        # Store a fact
        memory_id = await base_memory.store_memory(
            content="API requires authentication",
            category="fact",
            context="API integration"
        )
        
        # Search memories
        results = await base_memory.search_memories("API")
        print(f"Found {len(results)} relevant memories")
        
        # Get statistics
        stats = await base_memory.get_memory_stats()
        print(f"Memory statistics: {stats}")
        
        await base_memory.cleanup()
    
    await agent.start()
    print("Agent started with memory capabilities")
    
    # Agent will automatically use memory during conversations
    await asyncio.sleep(10)
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(memory_integration_example())
```

## Performance Considerations

### Memory Usage

- **Interaction Memory**: Minimal memory overhead, JSON-based storage
- **Agent Base Memory**: SQLite database, efficient indexing, query optimization

### Storage Limits

- **File System**: Consider disk space for database growth
- **Concurrent Access**: SQLite handles multiple connections efficiently
- **Query Performance**: Indexed searches scale well with database size

### Optimization Tips

1. **Use appropriate memory types** for different use cases
2. **Limit search results** to avoid performance issues
3. **Regular cleanup** of old or unused memories
4. **Monitor database size** and implement archiving strategies
5. **Use specific search queries** for better performance

## Error Handling

### Common Exceptions

```python
from spade_llm.memory.exceptions import (
    MemoryInitializationError,
    MemoryStorageError,
    MemorySearchError,
    MemoryBackendError
)

try:
    await memory.initialize()
    await memory.store_memory("content", "fact")
    results = await memory.search_memories("query")
except MemoryInitializationError as e:
    print(f"Failed to initialize memory: {e}")
except MemoryStorageError as e:
    print(f"Failed to store memory: {e}")
except MemorySearchError as e:
    print(f"Search failed: {e}")
except MemoryBackendError as e:
    print(f"Backend error: {e}")
```