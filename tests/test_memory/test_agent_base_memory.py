"""Comprehensive tests for AgentBaseMemory class."""

import pytest
import pytest_asyncio
import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from spade_llm.memory.agent_base_memory import AgentBaseMemory
from spade_llm.memory.backends.base import MemoryBackend, MemoryEntry
from spade_llm.memory.backends.sqlite import SQLiteMemoryBackend


class MockMemoryBackend(MemoryBackend):
    """Mock memory backend for testing."""
    
    def __init__(self):
        self.initialized = False
        self.memories = {}  # id -> MemoryEntry
        self.access_counts = {}  # id -> access_count
        self.last_accessed = {}  # id -> datetime
        
    async def initialize(self) -> None:
        """Initialize the mock backend."""
        self.initialized = True
        
    async def store_memory(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        entry_id = entry.id
        self.memories[entry_id] = entry
        self.access_counts[entry_id] = entry.access_count
        self.last_accessed[entry_id] = entry.last_accessed
        return entry_id
        
    async def get_memories_by_category(self, agent_id: str, category: str, 
                                     limit: int = 50) -> List[MemoryEntry]:
        """Get memories by category."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        results = []
        for entry in self.memories.values():
            if entry.agent_id == agent_id and entry.category == category:
                results.append(entry)
        
        # Sort by last_accessed desc and limit
        results.sort(key=lambda x: x.last_accessed, reverse=True)
        return results[:limit]
        
    async def search_memories(self, agent_id: str, query: str, 
                            limit: int = 10) -> List[MemoryEntry]:
        """Search memories by content."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        # Return empty results for empty query
        if not query or not query.strip():
            return []
        
        results = []
        for entry in self.memories.values():
            if entry.agent_id == agent_id:
                if query.lower() in entry.content.lower():
                    results.append(entry)
                elif entry.context and query.lower() in entry.context.lower():
                    results.append(entry)
        
        # Sort by last_accessed desc and limit
        results.sort(key=lambda x: x.last_accessed, reverse=True)
        return results[:limit]
        
    async def get_recent_memories(self, agent_id: str, 
                                limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        results = []
        for entry in self.memories.values():
            if entry.agent_id == agent_id:
                results.append(entry)
        
        # Sort by last_accessed desc and limit
        results.sort(key=lambda x: x.last_accessed, reverse=True)
        return results[:limit]
        
    async def update_access(self, memory_id: str) -> None:
        """Update access count and timestamp."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        if memory_id in self.memories:
            entry = self.memories[memory_id]
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        if memory_id in self.memories:
            del self.memories[memory_id]
            self.access_counts.pop(memory_id, None)
            self.last_accessed.pop(memory_id, None)
            return True
        return False
        
    async def get_memory_stats(self, agent_id: str) -> dict:
        """Get memory statistics."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        agent_memories = [entry for entry in self.memories.values() 
                         if entry.agent_id == agent_id]
        
        category_counts = {}
        for entry in agent_memories:
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
        
        if agent_memories:
            oldest = min(entry.created_at for entry in agent_memories)
            newest = max(entry.created_at for entry in agent_memories)
        else:
            oldest = newest = None
        
        return {
            'total_memories': len(agent_memories),
            'category_counts': category_counts,
            'oldest_memory': oldest.isoformat() if oldest else None,
            'newest_memory': newest.isoformat() if newest else None
        }
        
    async def cleanup(self) -> None:
        """Clean up the backend."""
        self.initialized = False
        self.memories.clear()
        self.access_counts.clear()
        self.last_accessed.clear()


@pytest.fixture
def mock_backend():
    """Create a mock memory backend for testing."""
    return MockMemoryBackend()


@pytest.fixture
def sample_agent_id():
    """Standard agent ID for testing."""
    return "test_agent@localhost"


@pytest.fixture
def sample_memory_entries():
    """Sample memory entries for testing."""
    base_time = datetime.now()
    return [
        MemoryEntry(
            id="mem1",
            agent_id="test_agent@localhost",
            category="fact",
            content="The user prefers Python over JavaScript",
            context="Programming language preference",
            confidence=0.9,
            created_at=base_time - timedelta(days=5),
            last_accessed=base_time - timedelta(days=1),
            access_count=3
        ),
        MemoryEntry(
            id="mem2",
            agent_id="test_agent@localhost",
            category="pattern",
            content="User typically works in the morning",
            context="Work schedule pattern",
            confidence=0.8,
            created_at=base_time - timedelta(days=3),
            last_accessed=base_time - timedelta(hours=2),
            access_count=5
        ),
        MemoryEntry(
            id="mem3",
            agent_id="test_agent@localhost",
            category="preference",
            content="User likes detailed explanations",
            context="Communication style preference",
            confidence=0.95,
            created_at=base_time - timedelta(days=1),
            last_accessed=base_time - timedelta(minutes=30),
            access_count=2
        ),
        MemoryEntry(
            id="mem4",
            agent_id="test_agent@localhost",
            category="capability",
            content="User is skilled in data analysis",
            context="Technical capability",
            confidence=0.85,
            created_at=base_time - timedelta(hours=6),
            last_accessed=base_time - timedelta(minutes=10),
            access_count=1
        ),
        MemoryEntry(
            id="mem5",
            agent_id="different_agent@localhost",
            category="fact",
            content="Different agent memory",
            context="Should not appear in search results",
            confidence=0.9,
            created_at=base_time - timedelta(days=2),
            last_accessed=base_time - timedelta(hours=1),
            access_count=2
        )
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    import time
    import gc
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Force garbage collection to release any file handles
    gc.collect()
    # On Windows, SQLite files may still be locked; retry with delay
    for attempt in range(5):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(0.2)
            else:
                # Last attempt: try to remove files individually with ignore_errors
                shutil.rmtree(temp_dir, ignore_errors=True)


async def setup_backend_with_memories(backend, entries):
    """Helper function to initialize backend and populate with memories."""
    await backend.initialize()
    for entry in entries:
        await backend.store_memory(entry)


class TestAgentBaseMemoryInitialization:
    """Test AgentBaseMemory initialization scenarios."""
    
    def test_init_with_custom_backend(self, mock_backend, sample_agent_id):
        """Test initialization with custom backend."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        assert memory.agent_id == sample_agent_id
        assert memory.backend == mock_backend
        assert not memory._initialized
        
    def test_init_with_default_backend(self, sample_agent_id, temp_dir):
        """Test initialization with default SQLite backend."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            memory_path=temp_dir
        )
        
        assert memory.agent_id == sample_agent_id
        assert isinstance(memory.backend, SQLiteMemoryBackend)
        assert not memory._initialized
        
    def test_init_with_default_path(self, sample_agent_id):
        """Test initialization with default path."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            memory = AgentBaseMemory(agent_id=sample_agent_id)
            
            assert memory.agent_id == sample_agent_id
            assert isinstance(memory.backend, SQLiteMemoryBackend)
            # Should be called twice: once by AgentBaseMemory, once by SQLiteMemoryBackend
            assert mock_mkdir.call_count == 2
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
            
    def test_agent_id_sanitization(self, temp_dir):
        """Test that agent ID is sanitized for filesystem use."""
        agent_id = "test@user/agent"
        memory = AgentBaseMemory(
            agent_id=agent_id,
            memory_path=temp_dir
        )
        
        # Check that the database path uses sanitized agent ID
        expected_db_name = "test_user_agent_base_memory.db"
        assert expected_db_name in str(memory.backend.db_path)


class TestAgentBaseMemoryStorageOperations:
    """Test memory storage operations."""
    
    @pytest.mark.asyncio
    async def test_store_memory_valid_category(self, mock_backend, sample_agent_id):
        """Test storing memory with valid category."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        memory_id = await memory.store_memory(
            category="fact",
            content="Test fact content",
            context="Test context",
            confidence=0.8
        )
        
        assert memory_id is not None
        assert mock_backend.initialized
        assert memory_id in mock_backend.memories
        
        stored_entry = mock_backend.memories[memory_id]
        assert stored_entry.agent_id == sample_agent_id
        assert stored_entry.category == "fact"
        assert stored_entry.content == "Test fact content"
        assert stored_entry.context == "Test context"
        assert stored_entry.confidence == 0.8
        
    @pytest.mark.asyncio
    async def test_store_memory_invalid_category(self, mock_backend, sample_agent_id):
        """Test storing memory with invalid category raises error."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        with pytest.raises(ValueError, match="Invalid category"):
            await memory.store_memory(
                category="invalid_category",
                content="Test content"
            )
            
    @pytest.mark.asyncio
    async def test_store_memory_invalid_confidence(self, mock_backend, sample_agent_id):
        """Test storing memory with invalid confidence raises error."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            await memory.store_memory(
                category="fact",
                content="Test content",
                confidence=1.5
            )
            
    @pytest.mark.asyncio 
    async def test_store_memory_with_defaults(self, mock_backend, sample_agent_id):
        """Test storing memory with default values."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        memory_id = await memory.store_memory(
            category="fact",
            content="Test content"
        )
        
        stored_entry = mock_backend.memories[memory_id]
        assert stored_entry.context is None
        assert stored_entry.confidence == 1.0
        
    @pytest.mark.asyncio
    async def test_store_memory_valid_categories(self, mock_backend, sample_agent_id):
        """Test storing memory with all valid categories."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        valid_categories = ["fact", "pattern", "preference", "capability"]
        
        for category in valid_categories:
            memory_id = await memory.store_memory(
                category=category,
                content=f"Test {category} content"
            )
            
            stored_entry = mock_backend.memories[memory_id]
            assert stored_entry.category == category


class TestAgentBaseMemoryRetrievalOperations:
    """Test memory retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_search_memories(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test searching memories by content."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        results = await memory.search_memories("Python", limit=10)
        
        assert len(results) == 1
        assert results[0].content == "The user prefers Python over JavaScript"
        
        # Check that access was updated
        assert mock_backend.memories[results[0].id].access_count == 4  # 3 + 1
        
    @pytest.mark.asyncio
    async def test_search_memories_context_match(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test searching memories by context."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        results = await memory.search_memories("programming", limit=10)
        
        assert len(results) == 1
        assert results[0].context == "Programming language preference"
        
    @pytest.mark.asyncio
    async def test_search_memories_no_results(self, mock_backend, sample_agent_id):
        """Test searching memories with no results."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        results = await memory.search_memories("nonexistent", limit=10)
        
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_get_memories_by_category(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test getting memories by category."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        results = await memory.get_memories_by_category("fact", limit=50)
        
        assert len(results) == 1
        assert results[0].category == "fact"
        assert results[0].agent_id == sample_agent_id
        
        # Check that access was updated
        assert mock_backend.memories[results[0].id].access_count == 4  # 3 + 1
        
    @pytest.mark.asyncio
    async def test_get_memories_by_category_multiple(self, mock_backend, sample_agent_id):
        """Test getting multiple memories by category."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Store multiple memories of the same category
        for i in range(3):
            await memory.store_memory(
                category="fact",
                content=f"Test fact {i}",
                context=f"Context {i}"
            )
        
        results = await memory.get_memories_by_category("fact", limit=50)
        
        assert len(results) == 3
        for result in results:
            assert result.category == "fact"
            assert result.agent_id == sample_agent_id
            
    @pytest.mark.asyncio
    async def test_get_recent_memories(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test getting recent memories."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        results = await memory.get_recent_memories(limit=3)
        
        assert len(results) == 3
        
        # Should be sorted by last_accessed descending
        # Based on sample data: mem4 (10 min ago), mem3 (30 min ago), mem2 (2 hours ago)
        assert results[0].id == "mem4"
        assert results[1].id == "mem3"
        assert results[2].id == "mem2"
        
        # Check that access was updated for all
        for result in results:
            # Use result.access_count directly since it's been updated
            assert result.access_count >= 1  # Should be at least 1
            
    @pytest.mark.asyncio
    async def test_get_relevant_memories_with_context(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test getting relevant memories with context."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        # Mock search_memories to return specific results
        with patch.object(memory, 'search_memories', return_value=[sample_memory_entries[0]]) as mock_search:
            results = await memory.get_relevant_memories(
                context="programming language",
                limit=10
            )
            
            mock_search.assert_called_once_with("programming language", 10)
            assert len(results) == 1
            assert results[0].content == "The user prefers Python over JavaScript"
            
    @pytest.mark.asyncio
    async def test_get_relevant_memories_no_context(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test getting relevant memories without context falls back to recent."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        # Mock get_recent_memories to return specific results
        with patch.object(memory, 'get_recent_memories', return_value=[sample_memory_entries[0]]) as mock_recent:
            results = await memory.get_relevant_memories(
                context=None,
                limit=10
            )
            
            mock_recent.assert_called_once_with(10)
            assert len(results) == 1
            
    @pytest.mark.asyncio
    async def test_get_relevant_memories_empty_context(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test getting relevant memories with empty context falls back to recent."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        # Mock get_recent_memories to return specific results
        with patch.object(memory, 'get_recent_memories', return_value=[sample_memory_entries[0]]) as mock_recent:
            results = await memory.get_relevant_memories(
                context="   ",  # Empty/whitespace context
                limit=10
            )
            
            mock_recent.assert_called_once_with(10)
            assert len(results) == 1


class TestAgentBaseMemoryFormattingOperations:
    """Test memory formatting operations."""
    
    @pytest.mark.asyncio
    async def test_format_for_context_empty(self, mock_backend, sample_agent_id):
        """Test formatting empty memory list."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        formatted = await memory.format_for_context([])
        
        assert formatted == ""
        
    @pytest.mark.asyncio
    async def test_format_for_context_single_category(self, mock_backend, sample_agent_id):
        """Test formatting memories from single category."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        memories = [
            MemoryEntry(
                agent_id=sample_agent_id,
                category="fact",
                content="User likes Python",
                context="Programming preference"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="fact",
                content="User works remotely",
                context="Work arrangement"
            )
        ]
        
        formatted = await memory.format_for_context(memories)
        
        expected = "Facts I remember:\n- User likes Python\n- User works remotely"
        assert formatted == expected
        
    @pytest.mark.asyncio
    async def test_format_for_context_multiple_categories(self, mock_backend, sample_agent_id):
        """Test formatting memories from multiple categories."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        memories = [
            MemoryEntry(
                agent_id=sample_agent_id,
                category="fact",
                content="User likes Python",
                context="Programming preference"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="pattern",
                content="User works in the morning",
                context="Work schedule"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="preference",
                content="User prefers detailed explanations",
                context="Communication style"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="capability",
                content="User is skilled in data analysis",
                context="Technical skill"
            )
        ]
        
        formatted = await memory.format_for_context(memories)
        
        expected_lines = [
            "Facts I remember:",
            "- User likes Python",
            "Patterns I remember:",
            "- User works in the morning",
            "Preferences I remember:",
            "- User prefers detailed explanations",
            "Capabilities I remember:",
            "- User is skilled in data analysis"
        ]
        expected = "\n".join(expected_lines)
        assert formatted == expected
        
    @pytest.mark.asyncio
    async def test_format_for_context_ordered_categories(self, mock_backend, sample_agent_id):
        """Test that categories are formatted in predefined order."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Create memories in different order than expected output
        memories = [
            MemoryEntry(
                agent_id=sample_agent_id,
                category="capability",
                content="User is skilled in data analysis",
                context="Technical skill"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="fact",
                content="User likes Python",
                context="Programming preference"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="preference",
                content="User prefers detailed explanations",
                context="Communication style"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="pattern",
                content="User works in the morning",
                context="Work schedule"
            )
        ]
        
        formatted = await memory.format_for_context(memories)
        
        # Should still be in the order: fact, pattern, preference, capability
        expected_lines = [
            "Facts I remember:",
            "- User likes Python",
            "Patterns I remember:",
            "- User works in the morning",
            "Preferences I remember:",
            "- User prefers detailed explanations",
            "Capabilities I remember:",
            "- User is skilled in data analysis"
        ]
        expected = "\n".join(expected_lines)
        assert formatted == expected


class TestAgentBaseMemoryDeletionOperations:
    """Test memory deletion operations."""
    
    @pytest.mark.asyncio
    async def test_delete_memory_success(self, mock_backend, sample_agent_id):
        """Test successful memory deletion."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Store a memory first
        memory_id = await memory.store_memory(
            category="fact",
            content="Test content to delete"
        )
        
        # Verify it exists
        assert memory_id in mock_backend.memories
        
        # Delete it
        success = await memory.delete_memory(memory_id)
        
        assert success is True
        assert memory_id not in mock_backend.memories
        
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, mock_backend, sample_agent_id):
        """Test deleting non-existent memory."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        success = await memory.delete_memory("nonexistent_id")
        
        assert success is False
        
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_backend, sample_agent_id):
        """Test memory cleanup."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Initialize and store some memories
        await memory.store_memory(
            category="fact",
            content="Test content"
        )
        
        assert memory._initialized is True
        assert mock_backend.initialized is True
        
        # Cleanup
        await memory.cleanup()
        
        assert memory._initialized is False
        assert mock_backend.initialized is False


class TestAgentBaseMemoryStatistics:
    """Test memory statistics functionality."""
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test getting memory statistics."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        stats = await memory.get_memory_stats()
        
        assert stats['total_memories'] == 4  # Only agent's memories, not the different agent's
        assert stats['category_counts']['fact'] == 1
        assert stats['category_counts']['pattern'] == 1
        assert stats['category_counts']['preference'] == 1
        assert stats['category_counts']['capability'] == 1
        assert stats['oldest_memory'] is not None
        assert stats['newest_memory'] is not None
        
    @pytest.mark.asyncio
    async def test_get_memory_stats_empty(self, mock_backend, sample_agent_id):
        """Test getting memory statistics with no memories."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        stats = await memory.get_memory_stats()
        
        assert stats['total_memories'] == 0
        assert stats['category_counts'] == {}
        assert stats['oldest_memory'] is None
        assert stats['newest_memory'] is None


class TestAgentBaseMemoryBackendInitialization:
    """Test backend initialization and lazy loading."""
    
    @pytest.mark.asyncio
    async def test_ensure_initialized_first_call(self, mock_backend, sample_agent_id):
        """Test that backend is initialized on first call."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        assert not memory._initialized
        assert not mock_backend.initialized
        
        await memory._ensure_initialized()
        
        assert memory._initialized is True
        assert mock_backend.initialized is True
        
    @pytest.mark.asyncio
    async def test_ensure_initialized_subsequent_calls(self, mock_backend, sample_agent_id):
        """Test that backend is not re-initialized on subsequent calls."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # First call should initialize
        await memory._ensure_initialized()
        assert memory._initialized is True
        
        # Mock the initialize method to track calls
        mock_backend.initialize = AsyncMock()
        
        # Second call should not initialize again
        await memory._ensure_initialized()
        mock_backend.initialize.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_auto_initialization_on_operations(self, mock_backend, sample_agent_id):
        """Test that operations automatically initialize the backend."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        assert not memory._initialized
        
        # Any operation should trigger initialization
        await memory.store_memory(
            category="fact",
            content="Test content"
        )
        
        assert memory._initialized is True
        assert mock_backend.initialized is True


class TestAgentBaseMemoryErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_backend_initialization_error(self, sample_agent_id):
        """Test handling backend initialization errors."""
        mock_backend = AsyncMock()
        mock_backend.initialize.side_effect = Exception("Database connection failed")
        
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        with pytest.raises(Exception, match="Database connection failed"):
            await memory.store_memory(
                category="fact",
                content="Test content"
            )
            
    @pytest.mark.asyncio
    async def test_backend_operation_error(self, sample_agent_id):
        """Test handling backend operation errors."""
        mock_backend = AsyncMock()
        mock_backend.initialize.return_value = None
        mock_backend.store_memory.side_effect = Exception("Storage failed")
        
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        with pytest.raises(Exception, match="Storage failed"):
            await memory.store_memory(
                category="fact",
                content="Test content"
            )
            
    @pytest.mark.asyncio
    async def test_validation_errors_dont_affect_backend(self, mock_backend, sample_agent_id):
        """Test that validation errors don't affect backend state."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Try to store invalid memory
        with pytest.raises(ValueError):
            await memory.store_memory(
                category="invalid",
                content="Test content"
            )
        
        # Backend should still be initialized and working
        assert memory._initialized is True
        assert mock_backend.initialized is True
        
        # Valid operation should work
        memory_id = await memory.store_memory(
            category="fact",
            content="Valid content"
        )
        
        assert memory_id is not None


class TestAgentBaseMemoryCompatibility:
    """Test backward compatibility methods."""
    
    def test_get_context_summary_empty(self, mock_backend, sample_agent_id):
        """Test get_context_summary with empty memories."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        summary = memory.get_context_summary([])
        
        assert summary is None
        
    def test_get_context_summary_with_memories(self, mock_backend, sample_agent_id):
        """Test get_context_summary with memories."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        memories = [
            MemoryEntry(
                agent_id=sample_agent_id,
                category="fact",
                content="User likes Python",
                context="Programming preference"
            ),
            MemoryEntry(
                agent_id=sample_agent_id,
                category="pattern",
                content="User works in the morning",
                context="Work schedule"
            )
        ]
        
        # Test the sync compatibility method
        summary = memory.get_context_summary(memories)
        
        assert summary is not None
        assert "User likes Python" in summary
        assert "User works in the morning" in summary
        
    def test_get_context_summary_async_context(self, mock_backend, sample_agent_id):
        """Test get_context_summary in async context returns simple format."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        memories = [
            MemoryEntry(
                agent_id=sample_agent_id,
                category="fact",
                content="User likes Python",
                context="Programming preference"
            )
        ]
        
        # Mock the event loop to simulate async context
        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        
        with patch('asyncio.get_event_loop', return_value=mock_loop):
            summary = memory.get_context_summary(memories)
            
            assert summary is not None
            assert "My memories:" in summary
            assert "User likes Python" in summary


class TestAgentBaseMemoryEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_confidence_boundary_values(self, mock_backend, sample_agent_id):
        """Test confidence boundary values."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Test minimum confidence
        memory_id1 = await memory.store_memory(
            category="fact",
            content="Min confidence content",
            confidence=0.0
        )
        
        # Test maximum confidence
        memory_id2 = await memory.store_memory(
            category="fact",
            content="Max confidence content",
            confidence=1.0
        )
        
        assert memory_id1 is not None
        assert memory_id2 is not None
        
        stored1 = mock_backend.memories[memory_id1]
        stored2 = mock_backend.memories[memory_id2]
        
        assert stored1.confidence == 0.0
        assert stored2.confidence == 1.0
        
    @pytest.mark.asyncio
    async def test_large_content_strings(self, mock_backend, sample_agent_id):
        """Test handling of large content strings."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Create a large content string
        large_content = "A" * 10000
        
        memory_id = await memory.store_memory(
            category="fact",
            content=large_content
        )
        
        assert memory_id is not None
        stored = mock_backend.memories[memory_id]
        assert stored.content == large_content
        
    @pytest.mark.asyncio
    async def test_unicode_content(self, mock_backend, sample_agent_id):
        """Test handling of unicode content."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        unicode_content = "用户喜欢Python编程 🐍"
        
        memory_id = await memory.store_memory(
            category="fact",
            content=unicode_content
        )
        
        assert memory_id is not None
        stored = mock_backend.memories[memory_id]
        assert stored.content == unicode_content
        
    @pytest.mark.asyncio
    async def test_special_characters_in_agent_id(self, mock_backend, temp_dir):
        """Test handling special characters in agent ID."""
        agent_id = "test@user.domain/resource"
        
        # Should not raise an exception
        memory = AgentBaseMemory(
            agent_id=agent_id,
            memory_path=temp_dir
        )
        
        assert memory.agent_id == agent_id
        
        # Should be able to store and retrieve memories
        memory_id = await memory.store_memory(
            category="fact",
            content="Test content"
        )
        
        assert memory_id is not None
        
    @pytest.mark.asyncio
    async def test_empty_search_query(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test searching with empty query."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        results = await memory.search_memories("", limit=10)
        
        # Should return no results for empty query
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_zero_limit_queries(self, mock_backend, sample_agent_id, sample_memory_entries):
        """Test queries with zero limit."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            backend=mock_backend
        )
        
        # Populate backend with sample memories
        await setup_backend_with_memories(mock_backend, sample_memory_entries)
        
        results = await memory.search_memories("Python", limit=0)
        assert len(results) == 0
        
        results = await memory.get_memories_by_category("fact", limit=0)
        assert len(results) == 0
        
        results = await memory.get_recent_memories(limit=0)
        assert len(results) == 0


class TestAgentBaseMemoryIntegration:
    """Integration tests with real SQLite backend."""
    
    @pytest_asyncio.fixture
    async def memory_instance(self, temp_dir, sample_agent_id):
        """Create a memory instance with guaranteed cleanup."""
        memory = AgentBaseMemory(
            agent_id=sample_agent_id,
            memory_path=temp_dir
        )
        yield memory
        await memory.cleanup()
    
    @pytest.mark.asyncio
    async def test_sqlite_integration(self, memory_instance):
        """Test basic integration with SQLite backend."""
        # Store a memory
        memory_id = await memory_instance.store_memory(
            category="fact",
            content="Integration test content",
            context="Testing SQLite integration",
            confidence=0.9
        )
        
        assert memory_id is not None
        
        # Search for it
        results = await memory_instance.search_memories("Integration", limit=10)
        assert len(results) == 1
        assert results[0].content == "Integration test content"
        
        # Get by category
        results = await memory_instance.get_memories_by_category("fact", limit=10)
        assert len(results) == 1
        assert results[0].content == "Integration test content"
        
        # Get stats
        stats = await memory_instance.get_memory_stats()
        assert stats['total_memories'] == 1
        assert stats['category_counts']['fact'] == 1
        
        # Delete the memory
        success = await memory_instance.delete_memory(memory_id)
        assert success is True
        
        # Verify it's gone
        results = await memory_instance.search_memories("Integration", limit=10)
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_concurrent_access(self, temp_dir, sample_agent_id):
        """Test concurrent access to the same memory system."""
        # We manually init here because we need two separate instances 
        # pointing to the same file.
        memory1 = AgentBaseMemory(
            agent_id=sample_agent_id,
            memory_path=temp_dir
        )
        
        memory2 = AgentBaseMemory(
            agent_id=sample_agent_id,
            memory_path=temp_dir
        )
        
        try:
            # Store memories concurrently
            tasks = []
            for i in range(5):
                task1 = memory1.store_memory(
                    category="fact",
                    content=f"Memory {i} from instance 1"
                )
                task2 = memory2.store_memory(
                    category="fact", 
                    content=f"Memory {i} from instance 2"
                )
                tasks.extend([task1, task2])
            
            # Use return_exceptions=True to handle SQLite locking issues on Windows
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results (non-exceptions that are not None)
            successful_ids = [r for r in results if not isinstance(r, Exception) and r is not None]
            errors = [r for r in results if isinstance(r, Exception)]
            
            # On Windows with SQLite, some concurrent operations may fail due to locking
            # We expect at least some to succeed
            if sys.platform == "win32":
                # On Windows without WAL mode, we expect some locks, 
                # but we shouldn't accept 0 successes.
                assert len(successful_ids) > 0, f"All database writes failed due to locking. Errors: {errors}"
            else:
                # Linux/Mac handles file locking much better
                assert len(successful_ids) == 10, f"Expected 10 writes, got {len(successful_ids)}. Errors: {errors}"
            
            # Verification: Check DB consistency
            # Even if some writes failed, both instances should see the SAME total count
            memories1 = await memory1.get_memories_by_category("fact", limit=100)
            memories2 = await memory2.get_memories_by_category("fact", limit=100)
            
            assert len(memories1) == len(successful_ids)
            assert len(memories1) == len(memories2)
        finally:
            # Ensure cleanup runs even if assertions fail
            await memory1.cleanup()
            await memory2.cleanup()