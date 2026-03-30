"""Tests for agent base memory tools module."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from spade_llm.memory.agent_base_memory import AgentBaseMemory
from spade_llm.memory.agent_base_memory_tools import (
    AgentBaseMemoryListTool,
    AgentBaseMemorySearchTool,
    AgentBaseMemoryStoreTool,
    create_base_memory_tools,
)
from spade_llm.memory.backends.base import MemoryEntry
from spade_llm.tools import LLMTool


class TestAgentBaseMemoryStoreTool:
    """Test AgentBaseMemoryStoreTool class."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance."""
        mock_memory = Mock(spec=AgentBaseMemory)
        mock_memory.store_memory = AsyncMock()
        return mock_memory

    @pytest.fixture
    def store_tool(self, mock_base_memory):
        """Create AgentBaseMemoryStoreTool instance."""
        return AgentBaseMemoryStoreTool(mock_base_memory)

    def test_initialization(self, mock_base_memory):
        """Test tool initialization."""
        tool = AgentBaseMemoryStoreTool(mock_base_memory)

        assert tool.base_memory == mock_base_memory
        assert tool.name == "store_memory"
        assert "Store important information" in tool.description
        assert isinstance(tool.parameters, dict)
        assert tool.parameters["type"] == "object"

        # Test required parameters
        assert "category" in tool.parameters["properties"]
        assert "content" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["category", "content"]

        # Test optional parameters
        assert "context" in tool.parameters["properties"]
        assert "confidence" in tool.parameters["properties"]

        # Test category enum
        category_enum = tool.parameters["properties"]["category"]["enum"]
        assert set(category_enum) == {"fact", "pattern", "preference", "capability"}

        # Test confidence constraints
        confidence_props = tool.parameters["properties"]["confidence"]
        assert confidence_props["minimum"] == 0.0
        assert confidence_props["maximum"] == 1.0
        assert confidence_props["default"] == 1.0

    def test_inheritance(self, store_tool):
        """Test that tool inherits from LLMTool."""
        assert isinstance(store_tool, LLMTool)
        assert hasattr(store_tool, "to_dict")
        assert hasattr(store_tool, "func")

    @pytest.mark.asyncio
    async def test_store_memory_basic(self, store_tool, mock_base_memory):
        """Test basic memory storage."""
        mock_base_memory.store_memory.return_value = "memory_id_123"

        result = await store_tool._store_memory(category="fact", content="Test fact content")

        assert result == "Stored fact memory: Test fact content"
        mock_base_memory.store_memory.assert_called_once_with(
            category="fact", content="Test fact content", context=None, confidence=1.0
        )

    @pytest.mark.asyncio
    async def test_store_memory_with_all_parameters(self, store_tool, mock_base_memory):
        """Test memory storage with all parameters."""
        mock_base_memory.store_memory.return_value = "memory_id_456"

        result = await store_tool._store_memory(
            category="pattern",
            content="User preference for dark mode",
            context="User settings conversation",
            confidence=0.8,
        )

        assert result == "Stored pattern memory: User preference for dark mode"
        mock_base_memory.store_memory.assert_called_once_with(
            category="pattern",
            content="User preference for dark mode",
            context="User settings conversation",
            confidence=0.8,
        )

    @pytest.mark.asyncio
    async def test_store_memory_all_categories(self, store_tool, mock_base_memory):
        """Test memory storage for all valid categories."""
        mock_base_memory.store_memory.return_value = "memory_id"

        categories = ["fact", "pattern", "preference", "capability"]

        for category in categories:
            result = await store_tool._store_memory(category=category, content=f"Test {category} content")

            assert result == f"Stored {category} memory: Test {category} content"

        assert mock_base_memory.store_memory.call_count == len(categories)

    @pytest.mark.asyncio
    async def test_store_memory_exception_handling(self, store_tool, mock_base_memory):
        """Test exception handling in memory storage."""
        mock_base_memory.store_memory.side_effect = ValueError("Invalid memory data")

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            result = await store_tool._store_memory(category="fact", content="Test content")

        assert result == "Failed to store memory: Invalid memory data"
        mock_logger.error.assert_called_once()
        assert "Failed to store memory" in mock_logger.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_store_memory_generic_exception(self, store_tool, mock_base_memory):
        """Test generic exception handling."""
        mock_base_memory.store_memory.side_effect = Exception("Generic error")

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            result = await store_tool._store_memory(category="fact", content="Test content")

        assert result == "Failed to store memory: Generic error"
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_via_llm_tool(self, store_tool, mock_base_memory):
        """Test execution through LLMTool interface."""
        mock_base_memory.store_memory.return_value = "memory_id"

        result = await store_tool.execute(
            category="preference", content="User likes JSON format", context="API response format", confidence=0.9
        )

        assert result == "Stored preference memory: User likes JSON format"
        mock_base_memory.store_memory.assert_called_once_with(
            category="preference", content="User likes JSON format", context="API response format", confidence=0.9
        )


class TestAgentBaseMemorySearchTool:
    """Test AgentBaseMemorySearchTool class."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance."""
        mock_memory = Mock(spec=AgentBaseMemory)
        mock_memory.search_memories = AsyncMock()
        return mock_memory

    @pytest.fixture
    def search_tool(self, mock_base_memory):
        """Create AgentBaseMemorySearchTool instance."""
        return AgentBaseMemorySearchTool(mock_base_memory)

    @pytest.fixture
    def sample_memories(self):
        """Create sample memory entries for testing."""
        return [
            MemoryEntry(
                id="mem_1",
                agent_id="agent@test.com",
                category="fact",
                content="API uses JSON format",
                context="API documentation",
                confidence=0.9,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            ),
            MemoryEntry(
                id="mem_2",
                agent_id="agent@test.com",
                category="pattern",
                content="User prefers concise responses",
                context="User interaction pattern",
                confidence=0.8,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=2,
            ),
        ]

    def test_initialization(self, mock_base_memory):
        """Test tool initialization."""
        tool = AgentBaseMemorySearchTool(mock_base_memory)

        assert tool.base_memory == mock_base_memory
        assert tool.name == "search_memories"
        assert "Search my long-term memory" in tool.description
        assert isinstance(tool.parameters, dict)
        assert tool.parameters["type"] == "object"

        # Test required parameters
        assert "query" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

        # Test optional parameters
        assert "limit" in tool.parameters["properties"]
        limit_props = tool.parameters["properties"]["limit"]
        assert limit_props["minimum"] == 1
        assert limit_props["maximum"] == 20
        assert limit_props["default"] == 10

    def test_inheritance(self, search_tool):
        """Test that tool inherits from LLMTool."""
        assert isinstance(search_tool, LLMTool)
        assert hasattr(search_tool, "to_dict")
        assert hasattr(search_tool, "func")

    @pytest.mark.asyncio
    async def test_search_memories_basic(self, search_tool, mock_base_memory, sample_memories):
        """Test basic memory search."""
        mock_base_memory.search_memories.return_value = sample_memories

        result = await search_tool._search_memories(query="API")

        assert "Found 2 memories for 'API':" in result
        assert "- [fact] API uses JSON format" in result
        assert "- [pattern] User prefers concise responses" in result
        assert "Context: API documentation" in result
        assert "Context: User interaction pattern" in result

        mock_base_memory.search_memories.assert_called_once_with("API", 10)

    @pytest.mark.asyncio
    async def test_search_memories_with_limit(self, search_tool, mock_base_memory, sample_memories):
        """Test memory search with custom limit."""
        mock_base_memory.search_memories.return_value = sample_memories[:1]

        result = await search_tool._search_memories(query="JSON", limit=5)

        assert "Found 1 memories for 'JSON':" in result
        assert "- [fact] API uses JSON format" in result

        mock_base_memory.search_memories.assert_called_once_with("JSON", 5)

    @pytest.mark.asyncio
    async def test_search_memories_no_results(self, search_tool, mock_base_memory):
        """Test memory search with no results."""
        mock_base_memory.search_memories.return_value = []

        result = await search_tool._search_memories(query="nonexistent")

        assert result == "No memories found for query: nonexistent"
        mock_base_memory.search_memories.assert_called_once_with("nonexistent", 10)

    @pytest.mark.asyncio
    async def test_search_memories_without_context(self, search_tool, mock_base_memory):
        """Test memory search with memories that have no context."""
        memories = [
            MemoryEntry(
                id="mem_1",
                agent_id="agent@test.com",
                category="fact",
                content="Simple fact",
                context=None,
                confidence=1.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            )
        ]
        mock_base_memory.search_memories.return_value = memories

        result = await search_tool._search_memories(query="fact")

        assert "Found 1 memories for 'fact':" in result
        assert "- [fact] Simple fact" in result
        assert "Context:" not in result  # No context line should be added

    @pytest.mark.asyncio
    async def test_search_memories_exception_handling(self, search_tool, mock_base_memory):
        """Test exception handling in memory search."""
        mock_base_memory.search_memories.side_effect = ValueError("Search failed")

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            result = await search_tool._search_memories(query="test")

        assert result == "Failed to search memories: Search failed"
        mock_logger.error.assert_called_once()
        assert "Failed to search memories" in mock_logger.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_memories_generic_exception(self, search_tool, mock_base_memory):
        """Test generic exception handling."""
        mock_base_memory.search_memories.side_effect = Exception("Generic error")

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            result = await search_tool._search_memories(query="test")

        assert result == "Failed to search memories: Generic error"
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_via_llm_tool(self, search_tool, mock_base_memory, sample_memories):
        """Test execution through LLMTool interface."""
        mock_base_memory.search_memories.return_value = sample_memories

        result = await search_tool.execute(query="test query", limit=15)

        assert "Found 2 memories for 'test query':" in result
        mock_base_memory.search_memories.assert_called_once_with("test query", 15)


class TestAgentBaseMemoryListTool:
    """Test AgentBaseMemoryListTool class."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance."""
        mock_memory = Mock(spec=AgentBaseMemory)
        mock_memory.get_memories_by_category = AsyncMock()
        return mock_memory

    @pytest.fixture
    def list_tool(self, mock_base_memory):
        """Create AgentBaseMemoryListTool instance."""
        return AgentBaseMemoryListTool(mock_base_memory)

    @pytest.fixture
    def sample_fact_memories(self):
        """Create sample fact memories for testing."""
        return [
            MemoryEntry(
                id="mem_1",
                agent_id="agent@test.com",
                category="fact",
                content="Python is a programming language",
                context="Programming discussion",
                confidence=1.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            ),
            MemoryEntry(
                id="mem_2",
                agent_id="agent@test.com",
                category="fact",
                content="HTTP status 200 means success",
                context=None,
                confidence=0.9,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            ),
        ]

    def test_initialization(self, mock_base_memory):
        """Test tool initialization."""
        tool = AgentBaseMemoryListTool(mock_base_memory)

        assert tool.base_memory == mock_base_memory
        assert tool.name == "list_memories"
        assert "List my memories organized by category" in tool.description
        assert isinstance(tool.parameters, dict)
        assert tool.parameters["type"] == "object"

        # Test required parameters
        assert "category" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["category"]

        # Test category enum
        category_enum = tool.parameters["properties"]["category"]["enum"]
        assert set(category_enum) == {"fact", "pattern", "preference", "capability"}

        # Test optional parameters
        assert "limit" in tool.parameters["properties"]
        limit_props = tool.parameters["properties"]["limit"]
        assert limit_props["minimum"] == 1
        assert limit_props["maximum"] == 50
        assert limit_props["default"] == 20

    def test_inheritance(self, list_tool):
        """Test that tool inherits from LLMTool."""
        assert isinstance(list_tool, LLMTool)
        assert hasattr(list_tool, "to_dict")
        assert hasattr(list_tool, "func")

    @pytest.mark.asyncio
    async def test_list_memories_basic(self, list_tool, mock_base_memory, sample_fact_memories):
        """Test basic memory listing."""
        mock_base_memory.get_memories_by_category.return_value = sample_fact_memories

        result = await list_tool._list_memories(category="fact")

        assert "My fact memories (2 total):" in result
        assert "- Python is a programming language" in result
        assert "- HTTP status 200 means success" in result
        assert "Context: Programming discussion" in result

        mock_base_memory.get_memories_by_category.assert_called_once_with("fact", 20)

    @pytest.mark.asyncio
    async def test_list_memories_with_limit(self, list_tool, mock_base_memory, sample_fact_memories):
        """Test memory listing with custom limit."""
        mock_base_memory.get_memories_by_category.return_value = sample_fact_memories[:1]

        result = await list_tool._list_memories(category="pattern", limit=5)

        assert "My pattern memories (1 total):" in result
        assert "- Python is a programming language" in result

        mock_base_memory.get_memories_by_category.assert_called_once_with("pattern", 5)

    @pytest.mark.asyncio
    async def test_list_memories_all_categories(self, list_tool, mock_base_memory):
        """Test memory listing for all valid categories."""
        mock_base_memory.get_memories_by_category.return_value = []

        categories = ["fact", "pattern", "preference", "capability"]

        for category in categories:
            result = await list_tool._list_memories(category=category)

            assert result == f"No {category} memories found."

        assert mock_base_memory.get_memories_by_category.call_count == len(categories)

    @pytest.mark.asyncio
    async def test_list_memories_no_results(self, list_tool, mock_base_memory):
        """Test memory listing with no results."""
        mock_base_memory.get_memories_by_category.return_value = []

        result = await list_tool._list_memories(category="capability")

        assert result == "No capability memories found."
        mock_base_memory.get_memories_by_category.assert_called_once_with("capability", 20)

    @pytest.mark.asyncio
    async def test_list_memories_without_context(self, list_tool, mock_base_memory):
        """Test memory listing with memories that have no context."""
        memories = [
            MemoryEntry(
                id="mem_1",
                agent_id="agent@test.com",
                category="preference",
                content="User prefers brief responses",
                context=None,
                confidence=1.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            )
        ]
        mock_base_memory.get_memories_by_category.return_value = memories

        result = await list_tool._list_memories(category="preference")

        assert "My preference memories (1 total):" in result
        assert "- User prefers brief responses" in result
        assert "Context:" not in result  # No context line should be added

    @pytest.mark.asyncio
    async def test_list_memories_exception_handling(self, list_tool, mock_base_memory):
        """Test exception handling in memory listing."""
        mock_base_memory.get_memories_by_category.side_effect = ValueError("List failed")

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            result = await list_tool._list_memories(category="fact")

        assert result == "Failed to list memories: List failed"
        mock_logger.error.assert_called_once()
        assert "Failed to list memories" in mock_logger.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_memories_generic_exception(self, list_tool, mock_base_memory):
        """Test generic exception handling."""
        mock_base_memory.get_memories_by_category.side_effect = Exception("Generic error")

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            result = await list_tool._list_memories(category="fact")

        assert result == "Failed to list memories: Generic error"
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_via_llm_tool(self, list_tool, mock_base_memory, sample_fact_memories):
        """Test execution through LLMTool interface."""
        mock_base_memory.get_memories_by_category.return_value = sample_fact_memories

        result = await list_tool.execute(category="fact", limit=30)

        assert "My fact memories (2 total):" in result
        mock_base_memory.get_memories_by_category.assert_called_once_with("fact", 30)


class TestCreateBaseMemoryTools:
    """Test create_base_memory_tools function."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance."""
        return Mock(spec=AgentBaseMemory)

    def test_create_base_memory_tools_basic(self, mock_base_memory):
        """Test basic tool creation."""
        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            tools = create_base_memory_tools(mock_base_memory)

        assert len(tools) == 3
        assert all(isinstance(tool, LLMTool) for tool in tools)

        # Check tool names
        tool_names = [tool.name for tool in tools]
        assert "store_memory" in tool_names
        assert "search_memories" in tool_names
        assert "list_memories" in tool_names

        # Check logging
        mock_logger.info.assert_called_once()
        assert "Created 3 base memory tools" in mock_logger.info.call_args[0][0]

    def test_create_base_memory_tools_types(self, mock_base_memory):
        """Test that created tools are the correct types."""
        tools = create_base_memory_tools(mock_base_memory)

        tool_types = [type(tool) for tool in tools]
        assert AgentBaseMemoryStoreTool in tool_types
        assert AgentBaseMemorySearchTool in tool_types
        assert AgentBaseMemoryListTool in tool_types

    def test_create_base_memory_tools_memory_reference(self, mock_base_memory):
        """Test that all tools have reference to the same memory instance."""
        tools = create_base_memory_tools(mock_base_memory)

        for tool in tools:
            assert tool.base_memory is mock_base_memory

    def test_create_base_memory_tools_immutability(self, mock_base_memory):
        """Test that returned tools list is independent."""
        tools1 = create_base_memory_tools(mock_base_memory)
        tools2 = create_base_memory_tools(mock_base_memory)

        assert tools1 is not tools2
        assert len(tools1) == len(tools2)

        # Tools should be different instances
        for i in range(len(tools1)):
            assert tools1[i] is not tools2[i]
            assert tools1[i].name == tools2[i].name

    def test_create_base_memory_tools_logging(self, mock_base_memory):
        """Test logging behavior."""
        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            tools = create_base_memory_tools(mock_base_memory)

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Created" in log_message
        assert "base memory tools" in log_message
        assert str(len(tools)) in log_message


class TestIntegrationScenarios:
    """Test integration scenarios with realistic usage patterns."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance with realistic behavior."""
        mock_memory = Mock(spec=AgentBaseMemory)
        mock_memory.store_memory = AsyncMock()
        mock_memory.search_memories = AsyncMock()
        mock_memory.get_memories_by_category = AsyncMock()
        return mock_memory

    @pytest.fixture
    def all_tools(self, mock_base_memory):
        """Create all memory tools."""
        return create_base_memory_tools(mock_base_memory)

    @pytest.fixture
    def sample_memories(self):
        """Create diverse sample memories."""
        return [
            MemoryEntry(
                id="mem_1",
                agent_id="agent@test.com",
                category="fact",
                content="API endpoint is https://api.example.com/v1",
                context="API documentation reading",
                confidence=1.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            ),
            MemoryEntry(
                id="mem_2",
                agent_id="agent@test.com",
                category="preference",
                content="User prefers CSV over JSON for data export",
                context="Data format discussion",
                confidence=0.8,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=3,
            ),
            MemoryEntry(
                id="mem_3",
                agent_id="agent@test.com",
                category="pattern",
                content="User typically asks for examples after explanations",
                context="Teaching interaction pattern",
                confidence=0.9,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=5,
            ),
        ]

    @pytest.mark.asyncio
    async def test_store_then_search_workflow(self, all_tools, mock_base_memory, sample_memories):
        """Test realistic workflow of storing then searching memories."""
        store_tool = next(tool for tool in all_tools if tool.name == "store_memory")
        search_tool = next(tool for tool in all_tools if tool.name == "search_memories")

        # Store a memory
        mock_base_memory.store_memory.return_value = "new_memory_id"
        store_result = await store_tool.execute(
            category="fact",
            content="New API uses OAuth2 authentication",
            context="Security documentation",
            confidence=0.95,
        )

        assert "Stored fact memory" in store_result
        mock_base_memory.store_memory.assert_called_once()

        # Search for related memories
        mock_base_memory.search_memories.return_value = [sample_memories[0]]
        search_result = await search_tool.execute(query="API", limit=5)

        assert "Found 1 memories for 'API'" in search_result
        assert "API endpoint is https://api.example.com/v1" in search_result
        mock_base_memory.search_memories.assert_called_once_with("API", 5)

    @pytest.mark.asyncio
    async def test_list_by_category_workflow(self, all_tools, mock_base_memory, sample_memories):
        """Test listing memories by different categories."""
        list_tool = next(tool for tool in all_tools if tool.name == "list_memories")

        # List facts
        fact_memories = [m for m in sample_memories if m.category == "fact"]
        mock_base_memory.get_memories_by_category.return_value = fact_memories

        result = await list_tool.execute(category="fact")

        assert "My fact memories (1 total):" in result
        assert "API endpoint is https://api.example.com/v1" in result
        mock_base_memory.get_memories_by_category.assert_called_with("fact", 20)

    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self, all_tools):
        """Test that tools properly validate their parameters."""
        store_tool = next(tool for tool in all_tools if tool.name == "store_memory")
        search_tool = next(tool for tool in all_tools if tool.name == "search_memories")
        list_tool = next(tool for tool in all_tools if tool.name == "list_memories")

        # Test parameter schemas
        store_schema = store_tool.parameters
        assert store_schema["properties"]["category"]["enum"] == ["fact", "pattern", "preference", "capability"]
        assert store_schema["properties"]["confidence"]["minimum"] == 0.0
        assert store_schema["properties"]["confidence"]["maximum"] == 1.0

        search_schema = search_tool.parameters
        assert search_schema["properties"]["limit"]["minimum"] == 1
        assert search_schema["properties"]["limit"]["maximum"] == 20

        list_schema = list_tool.parameters
        assert list_schema["properties"]["limit"]["minimum"] == 1
        assert list_schema["properties"]["limit"]["maximum"] == 50

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, all_tools, mock_base_memory, sample_memories):
        """Test concurrent execution of multiple tools."""
        store_tool = next(tool for tool in all_tools if tool.name == "store_memory")
        search_tool = next(tool for tool in all_tools if tool.name == "search_memories")
        list_tool = next(tool for tool in all_tools if tool.name == "list_memories")

        # Set up mock returns
        mock_base_memory.store_memory.return_value = "concurrent_memory_id"
        mock_base_memory.search_memories.return_value = sample_memories[:2]
        mock_base_memory.get_memories_by_category.return_value = sample_memories[2:]

        # Execute tools concurrently
        import asyncio

        store_task = store_tool.execute(category="fact", content="Concurrent test")
        search_task = search_tool.execute(query="test")
        list_task = list_tool.execute(category="pattern")

        store_result, search_result, list_result = await asyncio.gather(store_task, search_task, list_task)

        assert "Stored fact memory" in store_result
        assert "Found 2 memories for 'test'" in search_result
        assert "My pattern memories (1 total)" in list_result

        # Verify all backends were called
        mock_base_memory.store_memory.assert_called_once()
        mock_base_memory.search_memories.assert_called_once()
        mock_base_memory.get_memories_by_category.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_propagation_in_tools(self, all_tools, mock_base_memory):
        """Test that errors are properly handled and don't crash the system."""
        store_tool = next(tool for tool in all_tools if tool.name == "store_memory")
        search_tool = next(tool for tool in all_tools if tool.name == "search_memories")
        list_tool = next(tool for tool in all_tools if tool.name == "list_memories")

        # Set up different types of errors
        mock_base_memory.store_memory.side_effect = ValueError("Storage error")
        mock_base_memory.search_memories.side_effect = RuntimeError("Search error")
        mock_base_memory.get_memories_by_category.side_effect = ConnectionError("Connection error")

        # All tools should handle errors gracefully
        store_result = await store_tool.execute(category="fact", content="Test")
        search_result = await search_tool.execute(query="test")
        list_result = await list_tool.execute(category="fact")

        assert "Failed to store memory: Storage error" in store_result
        assert "Failed to search memories: Search error" in search_result
        assert "Failed to list memories: Connection error" in list_result

    def test_tool_serialization_for_llm_apis(self, all_tools):
        """Test that tools can be serialized for LLM API consumption."""
        for tool in all_tools:
            # Test to_dict method
            tool_dict = tool.to_dict()
            assert "name" in tool_dict
            assert "description" in tool_dict
            assert "parameters" in tool_dict

            # Test OpenAI format if available
            if hasattr(tool, "to_openai_tool"):
                openai_format = tool.to_openai_tool()
                assert "type" in openai_format
                assert "function" in openai_format
                assert openai_format["type"] == "function"
                assert openai_format["function"]["name"] == tool.name
