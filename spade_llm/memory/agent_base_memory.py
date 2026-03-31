"""Agent base memory implementation for long-term learning and knowledge storage."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends.base import MemoryBackend, MemoryEntry
from .backends.sqlite import SQLiteMemoryBackend

logger = logging.getLogger("spade_llm.memory.agent_base")


class AgentBaseMemory:
    """
    Agent base memory system for long-term learning and knowledge storage.

    This class provides persistent memory capabilities that allow agents to:
    - Store facts, patterns, preferences, and capabilities
    - Retrieve relevant memories for current conversations
    - Learn and improve over time across multiple conversations
    - Maintain knowledge that persists between agent restarts

    Unlike interaction memory which is conversation-specific, base memory
    is agent-specific and persists across all conversations.
    """

    def __init__(
        self,
        agent_id: str,
        backend: Optional[MemoryBackend] = None,
        memory_path: Optional[str] = None,
    ):
        """
        Initialize agent base memory.

        Args:
            agent_id: The JID of the agent owning this memory
            backend: Optional custom memory backend. If None, uses SQLite backend.
            memory_path: Optional custom memory storage path. If None, uses
                        default path "spade_llm/data/agent_memory".
        """
        self.agent_id = agent_id

        if backend is None:
            # Default to SQLite backend
            if memory_path is None:
                # Use default path relative to current working directory
                base_dir = Path("spade_llm/data/agent_memory")
            else:
                base_dir = Path(memory_path)

            # Ensure directory exists
            base_dir.mkdir(parents=True, exist_ok=True)

            # Create filename from agent_id (sanitize for filesystem)
            safe_agent_id = agent_id.replace("@", "_").replace("/", "_")
            db_path = base_dir / f"{safe_agent_id}_base_memory.db"
            backend = SQLiteMemoryBackend(str(db_path))

        self.backend = backend
        self._initialized = False

        logger.info(f"Initialized AgentBaseMemory for {agent_id}")

    async def _ensure_initialized(self):
        """Ensure the backend is initialized."""
        if not self._initialized:
            await self.backend.initialize()
            self._initialized = True

    async def store_memory(
        self,
        category: str,
        content: str,
        context: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """
        Store a memory entry.

        Args:
            category: Type of memory (fact, pattern, preference, capability)
            content: The memory content
            context: Optional context about this memory
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            The ID of the stored memory entry
        """
        await self._ensure_initialized()

        # Validate category
        valid_categories = ["fact", "pattern", "preference", "capability"]
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {valid_categories}")

        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        entry = MemoryEntry(
            agent_id=self.agent_id,
            category=category,
            content=content,
            context=context,
            confidence=confidence,
        )

        memory_id = await self.backend.store_memory(entry)
        logger.info(f"Stored {category} memory for {self.agent_id}: {content[:50]}...")
        return memory_id

    async def search_memories(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Search memories by content.

        Args:
            query: Search query string
            limit: Maximum number of memories to return

        Returns:
            List of memory entries matching the search query
        """
        await self._ensure_initialized()

        memories = await self.backend.search_memories(self.agent_id, query, limit)

        # Update access counts for retrieved memories
        for memory in memories:
            await self.backend.update_access(memory.id)

        logger.debug(f"Retrieved {len(memories)} memories for search query: {query}")
        return memories

    async def get_memories_by_category(self, category: str, limit: int = 50) -> List[MemoryEntry]:
        """
        Get memories by category.

        Args:
            category: The memory category to filter by
            limit: Maximum number of memories to return

        Returns:
            List of memory entries in the specified category
        """
        await self._ensure_initialized()

        memories = await self.backend.get_memories_by_category(self.agent_id, category, limit)

        # Update access counts for retrieved memories
        for memory in memories:
            await self.backend.update_access(memory.id)

        logger.debug(f"Retrieved {len(memories)} memories for category: {category}")
        return memories

    async def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """
        Get the most recently accessed memories.

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of recently accessed memory entries
        """
        await self._ensure_initialized()

        memories = await self.backend.get_recent_memories(self.agent_id, limit)

        # Update access counts for retrieved memories
        for memory in memories:
            await self.backend.update_access(memory.id)

        logger.debug(f"Retrieved {len(memories)} recent memories")
        return memories

    async def get_relevant_memories(self, context: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """
        Get memories relevant to the current context.

        This method combines different retrieval strategies to find the most
        relevant memories for the current conversation context.

        Args:
            context: Optional context string to search for relevant memories
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory entries
        """
        await self._ensure_initialized()

        if context and context.strip():
            # Search for memories matching the context
            memories = await self.search_memories(context, limit)
        else:
            # Fall back to recent memories
            memories = await self.get_recent_memories(limit)

        logger.debug(f"Retrieved {len(memories)} relevant memories for context")
        return memories

    async def format_for_context(self, memories: List[MemoryEntry]) -> str:
        """
        Format memories for injection into conversation context.

        Args:
            memories: List of memory entries to format

        Returns:
            Formatted string suitable for system message injection
        """
        if not memories:
            return ""

        # Group memories by category
        categories = {}
        for memory in memories:
            if memory.category not in categories:
                categories[memory.category] = []
            categories[memory.category].append(memory.content)

        # Format each category
        context_parts = []
        category_titles = {
            "fact": "Facts I remember",
            "pattern": "Patterns I remember",
            "preference": "Preferences I remember",
            "capability": "Capabilities I remember",
        }
        for category in ["fact", "pattern", "preference", "capability"]:
            if category in categories:
                category_title = category_titles[category]
                context_parts.append(f"{category_title}:")
                for content in categories[category]:
                    context_parts.append(f"- {content}")

        formatted = "\n".join(context_parts)
        logger.debug(f"Formatted {len(memories)} memories for context injection")
        return formatted

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if the memory was deleted, False if not found
        """
        await self._ensure_initialized()

        success = await self.backend.delete_memory(memory_id)
        if success:
            logger.info(f"Deleted memory {memory_id}")
        else:
            logger.warning(f"Memory {memory_id} not found for deletion")

        return success

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's memory usage.

        Returns:
            Dictionary with memory statistics
        """
        await self._ensure_initialized()

        stats = await self.backend.get_memory_stats(self.agent_id)
        logger.debug(f"Retrieved memory stats for {self.agent_id}: {stats}")
        return stats

    async def cleanup(self) -> None:
        """
        Clean up resources used by the memory system.

        This method should be called when the agent is shutting down
        to properly clean up database connections and other resources.
        """
        if self._initialized:
            await self.backend.cleanup()
            self._initialized = False
            logger.info(f"Cleaned up memory resources for {self.agent_id}")

    def get_context_summary(self, memories: List[MemoryEntry]) -> Optional[str]:
        """
        Get a formatted summary of memories for context injection.

        This method provides backward compatibility with the interaction memory
        interface while adapting it for base memory usage.

        Args:
            memories: List of memory entries to summarize

        Returns:
            Formatted summary string or None if no memories
        """
        if not memories:
            return None

        # Use the async format_for_context method synchronously
        # This is a compatibility method for non-async contexts
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use run_until_complete
            # Return a simple format instead
            summary = "My memories:\n"
            for memory in memories:
                summary += f"- {memory.content}\n"
            return summary.strip()
        else:
            # If we're not in an async context, we can use run_until_complete
            return loop.run_until_complete(self.format_for_context(memories))
