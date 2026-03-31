"""Abstract base class for memory backends."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry in the agent's base memory.

    Attributes:
        id: Unique identifier for the memory entry
        agent_id: JID of the agent that owns this memory
        category: Type of memory (fact, pattern, preference, capability)
        content: The actual memory content
        context: Optional context about when/why this memory was created
        confidence: Confidence score for this memory (0.0 to 1.0)
        created_at: When this memory was created
        last_accessed: When this memory was last accessed
        access_count: Number of times this memory has been accessed
    """

    agent_id: str
    category: str
    content: str
    context: Optional[str] = None
    confidence: float = 1.0
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    def __post_init__(self):
        """Set default values after initialization."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()


class MemoryBackend(ABC):
    """
    Abstract base class for memory storage backends.

    This interface defines the contract that all memory backends must implement,
    allowing for different storage implementations (SQLite, PostgreSQL, etc.)
    while maintaining a consistent API.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend storage.

        This method should create necessary tables, indexes, connections, etc.
        It's called once when the memory system starts up.
        """
        pass

    @abstractmethod
    async def store_memory(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in the backend.

        Args:
            entry: The memory entry to store

        Returns:
            The ID of the stored memory entry

        Raises:
            Exception: If storage fails
        """
        pass

    @abstractmethod
    async def get_memories_by_category(self, agent_id: str, category: str, limit: int = 50) -> List[MemoryEntry]:
        """
        Retrieve memories for an agent by category.

        Args:
            agent_id: The agent's JID
            category: The memory category to filter by
            limit: Maximum number of memories to return

        Returns:
            List of memory entries matching the criteria
        """
        pass

    @abstractmethod
    async def search_memories(self, agent_id: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Search memories by content.

        Args:
            agent_id: The agent's JID
            query: Search query string
            limit: Maximum number of memories to return

        Returns:
            List of memory entries matching the search query
        """
        pass

    @abstractmethod
    async def get_recent_memories(self, agent_id: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Get the most recently accessed memories for an agent.

        Args:
            agent_id: The agent's JID
            limit: Maximum number of memories to return

        Returns:
            List of recently accessed memory entries
        """
        pass

    @abstractmethod
    async def update_access(self, memory_id: str) -> None:
        """
        Update the access timestamp and count for a memory.

        Args:
            memory_id: The ID of the memory to update
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if the memory was deleted, False if not found
        """
        pass

    @abstractmethod
    async def get_memory_stats(self, agent_id: str) -> dict:
        """
        Get statistics about the agent's memory usage.

        Args:
            agent_id: The agent's JID

        Returns:
            Dictionary with memory statistics
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up resources used by the backend.

        This method should close connections, clean up temporary files, etc.
        It's called when the memory system shuts down.
        """
        pass
