"""SQLite backend implementation for agent base memory."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from .base import MemoryBackend, MemoryEntry

logger = logging.getLogger("spade_llm.memory.sqlite")


class SQLiteMemoryBackend(MemoryBackend):
    """
    SQLite-based memory backend using aiosqlite for async operations.

    This backend supports both persistent and in-memory storage:
    - File-based: Stores memories in a local SQLite database file for persistence
    - In-memory: Uses ":memory:" for temporary storage that is automatically
                 deleted when the agent stops (no persistence across restarts)

    Both modes provide good performance for typical agent memory usage.
    """

    def __init__(self, db_path: str):
        """
        Initialize the SQLite backend.

        Args:
            db_path: Path to the SQLite database file or ":memory:" for in-memory database

        Raises:
            ImportError: If aiosqlite is not installed
        """
        if aiosqlite is None:
            raise ImportError(
                "aiosqlite is required for SQLite backend. Install it with: pip install aiosqlite>=0.17.0"
            )

        # Handle in-memory database
        if db_path == ":memory:":
            self.db_path = ":memory:"
            self.is_in_memory = True
            self._connection = None  # Will hold persistent connection for in-memory DB
            logger.info("SQLite memory backend initialized with in-memory database")
        else:
            # Handle file-based database
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.is_in_memory = False
            self._connection = None
            logger.info(f"SQLite memory backend initialized with database: {self.db_path}")

        self._initialized = False

    async def _get_connection(self):
        """Get database connection - persistent for in-memory, new for file-based."""
        if self.is_in_memory:
            if self._connection is None:
                self._connection = await aiosqlite.connect(self.db_path)
            return self._connection
        else:
            # For file-based, return a new connection (to be used with async with)
            return aiosqlite.connect(self.db_path)

    async def _execute_with_connection(self, query_func):
        """Execute a function with the appropriate connection type."""
        if self.is_in_memory:
            # Use persistent connection for in-memory database
            db = await self._get_connection()
            return await query_func(db)
        else:
            # Use temporary connection for file-based database
            async with aiosqlite.connect(self.db_path) as db:
                return await query_func(db)

    async def initialize(self) -> None:
        """Initialize the SQLite database with required tables and indexes."""
        if self._initialized:
            return

        try:
            if self.is_in_memory:
                # For in-memory, use persistent connection
                db = await self._get_connection()
                # Create the main memories table
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_memories (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        content TEXT NOT NULL,
                        context TEXT,
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0
                    )
                """
                )

                # Create indexes for efficient queries
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_category
                    ON agent_memories(agent_id, category)
                """
                )

                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_recent
                    ON agent_memories(agent_id, last_accessed DESC)
                """
                )

                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_content
                    ON agent_memories(agent_id, content)
                """
                )

                await db.commit()
            else:
                # For file-based, use temporary connection
                async with aiosqlite.connect(self.db_path) as db:
                    # Create the main memories table
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_memories (
                            id TEXT PRIMARY KEY,
                            agent_id TEXT NOT NULL,
                            category TEXT NOT NULL,
                            content TEXT NOT NULL,
                            context TEXT,
                            confidence REAL DEFAULT 1.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            access_count INTEGER DEFAULT 0
                        )
                    """
                    )

                    # Create indexes for efficient queries
                    await db.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_category
                        ON agent_memories(agent_id, category)
                    """
                    )

                    await db.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_recent
                        ON agent_memories(agent_id, last_accessed DESC)
                    """
                    )

                    await db.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_content
                        ON agent_memories(agent_id, content)
                    """
                    )

                    await db.commit()

            self._initialized = True
            logger.info("SQLite memory backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SQLite memory backend: {e}")
            raise

    async def store_memory(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in the SQLite database.

        Args:
            entry: The memory entry to store

        Returns:
            The ID of the stored memory entry
        """
        await self.initialize()

        try:
            if self.is_in_memory:
                # Use persistent connection for in-memory database
                db = await self._get_connection()
                await db.execute(
                    """
                    INSERT OR REPLACE INTO agent_memories
                    (id, agent_id, category, content, context, confidence, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.id,
                        entry.agent_id,
                        entry.category,
                        entry.content,
                        entry.context,
                        entry.confidence,
                        (entry.created_at.isoformat() if entry.created_at else datetime.now().isoformat()),
                        (entry.last_accessed.isoformat() if entry.last_accessed else datetime.now().isoformat()),
                        entry.access_count,
                    ),
                )
                await db.commit()
            else:
                # Use temporary connection for file-based database
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO agent_memories
                        (id, agent_id, category, content, context, confidence, created_at, last_accessed, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            entry.id,
                            entry.agent_id,
                            entry.category,
                            entry.content,
                            entry.context,
                            entry.confidence,
                            (entry.created_at.isoformat() if entry.created_at else datetime.now().isoformat()),
                            (entry.last_accessed.isoformat() if entry.last_accessed else datetime.now().isoformat()),
                            entry.access_count,
                        ),
                    )
                    await db.commit()

            logger.debug(f"Stored memory {entry.id} for agent {entry.agent_id}")
            return entry.id

        except Exception as e:
            logger.error(f"Failed to store memory {entry.id}: {e}")
            raise

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
        await self.initialize()

        async def query_func(db):
            cursor = await db.execute(
                """
                SELECT id, agent_id, category, content, context, confidence,
                       created_at, last_accessed, access_count
                FROM agent_memories
                WHERE agent_id = ? AND category = ?
                ORDER BY last_accessed DESC
                LIMIT ?
            """,
                (agent_id, category, limit),
            )

            rows = await cursor.fetchall()
            memories = []

            for row in rows:
                memories.append(
                    MemoryEntry(
                        id=row[0],
                        agent_id=row[1],
                        category=row[2],
                        content=row[3],
                        context=row[4],
                        confidence=row[5],
                        created_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        last_accessed=(datetime.fromisoformat(row[7]) if row[7] else None),
                        access_count=row[8],
                    )
                )

            return memories

        try:
            memories = await self._execute_with_connection(query_func)
            logger.debug(f"Retrieved {len(memories)} memories for agent {agent_id}, category {category}")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories by category: {e}")
            raise

    async def search_memories(self, agent_id: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Search memories by content using SQLite's LIKE operator.

        Args:
            agent_id: The agent's JID
            query: Search query string
            limit: Maximum number of memories to return

        Returns:
            List of memory entries matching the search query
        """
        await self.initialize()

        async def query_func(db):
            # Use LIKE for simple text search
            search_pattern = f"%{query}%"

            cursor = await db.execute(
                """
                SELECT id, agent_id, category, content, context, confidence,
                       created_at, last_accessed, access_count
                FROM agent_memories
                WHERE agent_id = ? AND (content LIKE ? OR context LIKE ?)
                ORDER BY last_accessed DESC
                LIMIT ?
            """,
                (agent_id, search_pattern, search_pattern, limit),
            )

            rows = await cursor.fetchall()
            memories = []

            for row in rows:
                memories.append(
                    MemoryEntry(
                        id=row[0],
                        agent_id=row[1],
                        category=row[2],
                        content=row[3],
                        context=row[4],
                        confidence=row[5],
                        created_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        last_accessed=(datetime.fromisoformat(row[7]) if row[7] else None),
                        access_count=row[8],
                    )
                )

            return memories

        try:
            memories = await self._execute_with_connection(query_func)
            logger.debug(f"Found {len(memories)} memories for agent {agent_id} matching '{query}'")
            return memories

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise

    async def get_recent_memories(self, agent_id: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Get the most recently accessed memories for an agent.

        Args:
            agent_id: The agent's JID
            limit: Maximum number of memories to return

        Returns:
            List of recently accessed memory entries
        """
        await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT id, agent_id, category, content, context, confidence,
                           created_at, last_accessed, access_count
                    FROM agent_memories
                    WHERE agent_id = ?
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """,
                    (agent_id, limit),
                )

                rows = await cursor.fetchall()
                memories = []

                for row in rows:
                    memories.append(
                        MemoryEntry(
                            id=row[0],
                            agent_id=row[1],
                            category=row[2],
                            content=row[3],
                            context=row[4],
                            confidence=row[5],
                            created_at=(datetime.fromisoformat(row[6]) if row[6] else None),
                            last_accessed=(datetime.fromisoformat(row[7]) if row[7] else None),
                            access_count=row[8],
                        )
                    )

                logger.debug(f"Retrieved {len(memories)} recent memories for agent {agent_id}")
                return memories

        except Exception as e:
            logger.error(f"Failed to retrieve recent memories: {e}")
            raise

    async def update_access(self, memory_id: str) -> None:
        """
        Update the access timestamp and count for a memory.

        Args:
            memory_id: The ID of the memory to update
        """
        await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE agent_memories
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """,
                    (datetime.now().isoformat(), memory_id),
                )

                await db.commit()

            logger.debug(f"Updated access for memory {memory_id}")

        except Exception as e:
            logger.error(f"Failed to update access for memory {memory_id}: {e}")
            raise

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if the memory was deleted, False if not found
        """
        await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    DELETE FROM agent_memories WHERE id = ?
                """,
                    (memory_id,),
                )

                await db.commit()

                deleted = cursor.rowcount > 0
                logger.debug(f"Memory {memory_id} deleted: {deleted}")
                return deleted

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    async def get_memory_stats(self, agent_id: str) -> dict:
        """
        Get statistics about the agent's memory usage.

        Args:
            agent_id: The agent's JID

        Returns:
            Dictionary with memory statistics
        """
        await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get total count
                cursor = await db.execute(
                    """
                    SELECT COUNT(*) FROM agent_memories WHERE agent_id = ?
                """,
                    (agent_id,),
                )
                total_count = (await cursor.fetchone())[0]

                # Get count by category
                cursor = await db.execute(
                    """
                    SELECT category, COUNT(*) FROM agent_memories
                    WHERE agent_id = ? GROUP BY category
                """,
                    (agent_id,),
                )
                category_counts = {row[0]: row[1] for row in await cursor.fetchall()}

                # Get oldest and newest memories
                cursor = await db.execute(
                    """
                    SELECT MIN(created_at), MAX(created_at) FROM agent_memories
                    WHERE agent_id = ?
                """,
                    (agent_id,),
                )
                oldest, newest = await cursor.fetchone()

                stats = {
                    "total_memories": total_count,
                    "category_counts": category_counts,
                    "oldest_memory": oldest,
                    "newest_memory": newest,
                }

                logger.debug(f"Retrieved memory stats for agent {agent_id}: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            raise

    async def cleanup(self) -> None:
        """
        Clean up resources used by the SQLite backend.

        For file-based databases, this is mostly a no-op since connections are
        created and closed per operation. For in-memory databases, the database
        is automatically deleted when the last connection closes.
        """
        if self.is_in_memory and self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("SQLite memory backend cleanup completed (in-memory database automatically deleted)")
        else:
            logger.info("SQLite memory backend cleanup completed")
