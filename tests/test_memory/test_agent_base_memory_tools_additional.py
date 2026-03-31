"""Additional comprehensive tests for agent base memory tools module."""

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


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for agent base memory tools."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance."""
        mock_memory = Mock(spec=AgentBaseMemory)
        mock_memory.store_memory = AsyncMock()
        mock_memory.search_memories = AsyncMock()
        mock_memory.get_memories_by_category = AsyncMock()
        return mock_memory

    @pytest.fixture
    def all_tools(self, mock_base_memory):
        """Create all memory tools."""
        return create_base_memory_tools(mock_base_memory)

    @pytest.mark.asyncio
    async def test_store_memory_edge_cases(self, mock_base_memory):
        """Test store memory with edge case inputs."""
        tool = AgentBaseMemoryStoreTool(mock_base_memory)
        mock_base_memory.store_memory.return_value = "edge_case_id"

        # Test with empty strings
        result = await tool._store_memory(category="fact", content="", context="", confidence=0.0)
        assert "Stored fact memory:" in result
        mock_base_memory.store_memory.assert_called_with(category="fact", content="", context="", confidence=0.0)

        # Test with maximum confidence
        result = await tool._store_memory(category="capability", content="Max confidence test", confidence=1.0)
        assert "Stored capability memory: Max confidence test" in result

        # Test with very long content
        long_content = "x" * 1000
        result = await tool._store_memory(category="pattern", content=long_content, confidence=0.5)
        assert f"Stored pattern memory: {long_content}" in result

        # Test with special characters
        special_content = "Memory with 特殊字符 and émojis 🚀"
        result = await tool._store_memory(category="preference", content=special_content)
        assert f"Stored preference memory: {special_content}" in result

    @pytest.mark.asyncio
    async def test_search_memory_edge_cases(self, mock_base_memory):
        """Test search memory with edge case inputs."""
        tool = AgentBaseMemorySearchTool(mock_base_memory)

        # Test with minimum limit
        mock_base_memory.search_memories.return_value = []
        result = await tool._search_memories(query="test", limit=1)
        assert "No memories found for query: test" in result
        mock_base_memory.search_memories.assert_called_with("test", 1)

        # Test with maximum limit
        mock_base_memory.search_memories.return_value = []
        result = await tool._search_memories(query="test", limit=20)
        assert "No memories found for query: test" in result
        mock_base_memory.search_memories.assert_called_with("test", 20)

        # Test with empty query
        result = await tool._search_memories(query="", limit=5)
        assert "No memories found for query:" in result

        # Test with query containing special characters
        special_query = "特殊查询 with émojis 🔍"
        result = await tool._search_memories(query=special_query, limit=10)
        assert f"No memories found for query: {special_query}" in result

        # Test with very long query
        long_query = "x" * 500
        result = await tool._search_memories(query=long_query, limit=15)
        assert f"No memories found for query: {long_query}" in result

    @pytest.mark.asyncio
    async def test_list_memory_edge_cases(self, mock_base_memory):
        """Test list memory with edge case inputs."""
        tool = AgentBaseMemoryListTool(mock_base_memory)

        # Test with minimum limit
        mock_base_memory.get_memories_by_category.return_value = []
        result = await tool._list_memories(category="fact", limit=1)
        assert "No fact memories found." in result
        mock_base_memory.get_memories_by_category.assert_called_with("fact", 1)

        # Test with maximum limit
        mock_base_memory.get_memories_by_category.return_value = []
        result = await tool._list_memories(category="capability", limit=50)
        assert "No capability memories found." in result
        mock_base_memory.get_memories_by_category.assert_called_with("capability", 50)

        # Test with all category types
        categories = ["fact", "pattern", "preference", "capability"]
        for category in categories:
            mock_base_memory.get_memories_by_category.return_value = []
            result = await tool._list_memories(category=category, limit=10)
            assert f"No {category} memories found." in result

    @pytest.mark.asyncio
    async def test_memory_entries_with_edge_case_data(self, mock_base_memory):
        """Test tools with memory entries containing edge case data."""
        search_tool = AgentBaseMemorySearchTool(mock_base_memory)
        list_tool = AgentBaseMemoryListTool(mock_base_memory)

        # Create memory entries with edge case data
        edge_case_memories = [
            MemoryEntry(
                id="mem_empty",
                agent_id="agent@test.com",
                category="fact",
                content="",  # Empty content
                context="",  # Empty context
                confidence=0.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
            ),
            MemoryEntry(
                id="mem_special",
                agent_id="agent@test.com",
                category="pattern",
                content="Memory with 特殊字符 and émojis 🚀",
                context="Context with special chars ñáéíóú",
                confidence=1.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=999,
            ),
            MemoryEntry(
                id="mem_long",
                agent_id="agent@test.com",
                category="preference",
                content="x" * 1000,  # Very long content
                context="y" * 500,  # Very long context
                confidence=0.5,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            ),
        ]

        # Test search tool with edge case memories
        mock_base_memory.search_memories.return_value = edge_case_memories
        result = await search_tool._search_memories(query="test")

        assert "Found 3 memories for 'test':" in result
        assert "- [fact]" in result  # Empty content should still show
        assert "- [pattern] Memory with 特殊字符 and émojis 🚀" in result
        assert "- [preference] " + "x" * 1000 in result
        assert "Context: Context with special chars ñáéíóú" in result
        assert "Context: " + "y" * 500 in result

        # Test list tool with edge case memories
        mock_base_memory.get_memories_by_category.return_value = edge_case_memories
        result = await list_tool._list_memories(category="fact")

        assert "My fact memories (3 total):" in result
        assert "- " in result  # Empty content
        assert "- Memory with 特殊字符 and émojis 🚀" in result
        assert "- " + "x" * 1000 in result

    @pytest.mark.asyncio
    async def test_concurrent_access_to_same_memory(self, mock_base_memory):
        """Test concurrent access to the same memory instance."""
        import asyncio

        # Create multiple tools using the same memory instance
        tools = create_base_memory_tools(mock_base_memory)
        store_tool = next(t for t in tools if t.name == "store_memory")
        search_tool = next(t for t in tools if t.name == "search_memories")
        list_tool = next(t for t in tools if t.name == "list_memories")

        # Set up mock returns
        mock_base_memory.store_memory.return_value = "concurrent_id"
        mock_base_memory.search_memories.return_value = []
        mock_base_memory.get_memories_by_category.return_value = []

        # Execute many concurrent operations
        tasks = []
        for i in range(10):
            tasks.append(store_tool.execute(category="fact", content=f"Fact {i}"))
            tasks.append(search_tool.execute(query=f"query {i}"))
            tasks.append(list_tool.execute(category="pattern"))

        results = await asyncio.gather(*tasks)

        # Verify all operations completed
        assert len(results) == 30
        assert all(
            "Stored fact memory:" in result or "No memories found" in result or "No pattern memories" in result
            for result in results
        )

        # Verify backend was called the expected number of times
        assert mock_base_memory.store_memory.call_count == 10
        assert mock_base_memory.search_memories.call_count == 10
        assert mock_base_memory.get_memories_by_category.call_count == 10

    @pytest.mark.asyncio
    async def test_memory_backend_timeout_simulation(self, mock_base_memory):
        """Test tools behavior when memory backend times out."""
        import asyncio

        async def timeout_mock(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError("Memory backend timeout")

        mock_base_memory.store_memory.side_effect = timeout_mock
        mock_base_memory.search_memories.side_effect = timeout_mock
        mock_base_memory.get_memories_by_category.side_effect = timeout_mock

        store_tool = AgentBaseMemoryStoreTool(mock_base_memory)
        search_tool = AgentBaseMemorySearchTool(mock_base_memory)
        list_tool = AgentBaseMemoryListTool(mock_base_memory)

        # Test all tools handle timeouts gracefully
        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            store_result = await store_tool.execute(category="fact", content="Test")
            search_result = await search_tool.execute(query="test")
            list_result = await list_tool.execute(category="fact")

        assert "Failed to store memory: Memory backend timeout" in store_result
        assert "Failed to search memories: Memory backend timeout" in search_result
        assert "Failed to list memories: Memory backend timeout" in list_result

        # Verify logging happened
        assert mock_logger.error.call_count == 3

    def test_tool_parameter_schema_validation(self, mock_base_memory):
        """Test that tool parameter schemas are properly structured."""
        tools = create_base_memory_tools(mock_base_memory)

        for tool in tools:
            schema = tool.parameters

            # Verify basic schema structure
            assert "type" in schema
            assert "properties" in schema
            assert "required" in schema
            assert schema["type"] == "object"

            # Verify all required fields are in properties
            for required_field in schema["required"]:
                assert required_field in schema["properties"]

            # Verify each property has required fields
            for prop_name, prop_schema in schema["properties"].items():
                assert "type" in prop_schema
                assert "description" in prop_schema

                # Check enum constraints
                if "enum" in prop_schema:
                    assert isinstance(prop_schema["enum"], list)
                    assert len(prop_schema["enum"]) > 0

                # Check numeric constraints
                if prop_schema["type"] in ["integer", "number"]:
                    if "minimum" in prop_schema:
                        assert isinstance(prop_schema["minimum"], (int, float))
                    if "maximum" in prop_schema:
                        assert isinstance(prop_schema["maximum"], (int, float))
                    if "default" in prop_schema:
                        assert isinstance(prop_schema["default"], (int, float))

    def test_tool_names_and_descriptions(self, mock_base_memory):
        """Test that tool names and descriptions are appropriate."""
        tools = create_base_memory_tools(mock_base_memory)

        expected_names = {"store_memory", "search_memories", "list_memories"}
        actual_names = {tool.name for tool in tools}
        assert actual_names == expected_names

        for tool in tools:
            # Name should be lowercase with underscores
            assert tool.name.islower()
            assert " " not in tool.name

            # Description should be substantial
            assert len(tool.description) > 50
            assert "memor" in tool.description.lower()  # Matches "memory" or "memories"

            # Description should contain usage guidance
            assert "Use this tool" in tool.description

    @pytest.mark.asyncio
    async def test_tool_execution_with_various_parameter_combinations(self, mock_base_memory):
        """Test tools with various parameter combinations."""
        mock_base_memory.store_memory.return_value = "param_test_id"
        mock_base_memory.search_memories.return_value = []
        mock_base_memory.get_memories_by_category.return_value = []

        store_tool = AgentBaseMemoryStoreTool(mock_base_memory)
        search_tool = AgentBaseMemorySearchTool(mock_base_memory)
        list_tool = AgentBaseMemoryListTool(mock_base_memory)

        # Test store tool with various confidence levels
        confidence_levels = [0.0, 0.1, 0.5, 0.9, 1.0]
        for confidence in confidence_levels:
            await store_tool.execute(
                category="fact", content=f"Test with confidence {confidence}", confidence=confidence
            )

        # Test search tool with various limits
        limits = [1, 5, 10, 15, 20]
        for limit in limits:
            await search_tool.execute(query="test query", limit=limit)

        # Test list tool with various limits
        limits = [1, 10, 20, 30, 50]
        for limit in limits:
            await list_tool.execute(category="fact", limit=limit)

        # Verify all calls were made
        assert mock_base_memory.store_memory.call_count == len(confidence_levels)
        assert mock_base_memory.search_memories.call_count == len(limits)
        assert mock_base_memory.get_memories_by_category.call_count == len(limits)

    @pytest.mark.asyncio
    async def test_logging_behavior_detailed(self, mock_base_memory):
        """Test detailed logging behavior."""
        mock_base_memory.store_memory.side_effect = Exception("Storage error")
        mock_base_memory.search_memories.side_effect = Exception("Search error")
        mock_base_memory.get_memories_by_category.side_effect = Exception("List error")

        store_tool = AgentBaseMemoryStoreTool(mock_base_memory)
        search_tool = AgentBaseMemorySearchTool(mock_base_memory)
        list_tool = AgentBaseMemoryListTool(mock_base_memory)

        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            await store_tool.execute(category="fact", content="Test")
            await search_tool.execute(query="test")
            await list_tool.execute(category="fact")

            # Verify logging calls
            assert mock_logger.error.call_count == 3

            # Verify log messages contain expected information
            log_calls = mock_logger.error.call_args_list
            assert "Failed to store memory" in log_calls[0][0][0]
            assert "Failed to search memories" in log_calls[1][0][0]
            assert "Failed to list memories" in log_calls[2][0][0]

        # Test successful logging for create_base_memory_tools
        with patch("spade_llm.memory.agent_base_memory_tools.logger") as mock_logger:
            create_base_memory_tools(mock_base_memory)

            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "Created" in log_message
            assert "3" in log_message
            assert "base memory tools" in log_message


class TestRegressionAndCompatibility:
    """Test regression scenarios and compatibility."""

    @pytest.fixture
    def mock_base_memory(self):
        """Create a mock AgentBaseMemory instance."""
        mock_memory = Mock(spec=AgentBaseMemory)
        mock_memory.store_memory = AsyncMock()
        mock_memory.search_memories = AsyncMock()
        mock_memory.get_memories_by_category = AsyncMock()
        return mock_memory

    def test_tool_inheritance_chain(self, mock_base_memory):
        """Test that tools maintain proper inheritance chain."""
        tools = create_base_memory_tools(mock_base_memory)

        for tool in tools:
            # Test MRO (Method Resolution Order)
            mro = tool.__class__.__mro__
            assert LLMTool in mro
            assert object in mro

            # Test required methods exist
            assert hasattr(tool, "execute")
            assert hasattr(tool, "to_dict")
            assert callable(tool.execute)
            assert callable(tool.to_dict)

            # Test tool attributes
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "parameters")
            assert hasattr(tool, "func")

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_old_memory_entries(self, mock_base_memory):
        """Test compatibility with potentially old memory entry formats."""
        search_tool = AgentBaseMemorySearchTool(mock_base_memory)
        list_tool = AgentBaseMemoryListTool(mock_base_memory)

        # Create memory entries that might have different attribute combinations
        varied_memories = [
            MemoryEntry(
                id="mem_1",
                agent_id="agent@test.com",
                category="fact",
                content="Normal memory",
                context="Normal context",
                confidence=0.8,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            ),
            MemoryEntry(
                id="mem_2",
                agent_id="agent@test.com",
                category="pattern",
                content="Memory without context",
                context=None,  # Explicitly None
                confidence=1.0,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
            ),
        ]

        # Test search tool handles varied memory formats
        mock_base_memory.search_memories.return_value = varied_memories
        search_result = await search_tool._search_memories(query="test")

        assert "Found 2 memories for 'test':" in search_result
        assert "- [fact] Normal memory" in search_result
        assert "- [pattern] Memory without context" in search_result
        assert "Context: Normal context" in search_result
        # The None context should not create a context line

        # Test list tool handles varied memory formats
        mock_base_memory.get_memories_by_category.return_value = varied_memories
        list_result = await list_tool._list_memories(category="fact")

        assert "My fact memories (2 total):" in list_result
        assert "- Normal memory" in list_result
        assert "- Memory without context" in list_result

    def test_tool_serialization_formats(self, mock_base_memory):
        """Test that tools can be serialized in various formats."""
        tools = create_base_memory_tools(mock_base_memory)

        for tool in tools:
            # Test basic dict serialization
            tool_dict = tool.to_dict()
            assert isinstance(tool_dict, dict)
            assert "name" in tool_dict
            assert "description" in tool_dict
            assert "parameters" in tool_dict

            # Test that dict can be JSON serialized
            import json

            json_str = json.dumps(tool_dict)
            assert isinstance(json_str, str)

            # Test that JSON can be deserialized
            restored_dict = json.loads(json_str)
            assert restored_dict == tool_dict

            # Test OpenAI format if available
            if hasattr(tool, "to_openai_tool"):
                openai_format = tool.to_openai_tool()
                assert isinstance(openai_format, dict)
                assert "type" in openai_format
                assert "function" in openai_format
                assert openai_format["type"] == "function"

                # Test OpenAI format serialization
                openai_json = json.dumps(openai_format)
                assert isinstance(openai_json, str)

    def test_tool_factory_consistency(self, mock_base_memory):
        """Test that tool factory creates consistent tools."""
        # Create tools multiple times
        tools_1 = create_base_memory_tools(mock_base_memory)
        tools_2 = create_base_memory_tools(mock_base_memory)

        # Should create same number of tools
        assert len(tools_1) == len(tools_2)

        # Should have same names
        names_1 = {tool.name for tool in tools_1}
        names_2 = {tool.name for tool in tools_2}
        assert names_1 == names_2

        # Should have same types
        types_1 = {type(tool) for tool in tools_1}
        types_2 = {type(tool) for tool in tools_2}
        assert types_1 == types_2

        # Should reference same memory instance
        for tool in tools_1:
            assert tool.base_memory is mock_base_memory
        for tool in tools_2:
            assert tool.base_memory is mock_base_memory
