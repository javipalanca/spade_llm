"""Auto-registered tools for agent base memory functionality."""

import logging
from typing import List

from ..tools import LLMTool
from .agent_base_memory import AgentBaseMemory

logger = logging.getLogger("spade_llm.memory.tools")


class AgentBaseMemoryStoreTool(LLMTool):
    """
    Tool for storing memories in agent base memory.

    This tool allows the LLM to consciously decide when to store information
    for future use, enabling long-term learning and knowledge accumulation.
    """

    def __init__(self, base_memory: AgentBaseMemory):
        """
        Initialize the memory store tool.

        Args:
            base_memory: The agent base memory instance to use
        """
        self.base_memory = base_memory

        super().__init__(
            name="store_memory",
            description="""Store important information in my long-term memory for future use.
            Use this tool to remember:
            - Facts: Concrete information about APIs, data formats, configurations, etc.
            - Patterns: Recurring behaviors or trends I've observed
            - Preferences: User preferences or system settings I've learned
            - Capabilities: My own abilities or limitations I've discovered

            This memory persists across conversations and helps me improve over time.""",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["fact", "pattern", "preference", "capability"],
                        "description": "Type of memory: fact (concrete info), pattern (observed behavior), preference (learned preference), capability (my ability/limitation)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The information to remember. Be specific and concise.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about when/why this is important",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 1.0,
                        "description": "Confidence level in this memory (0.0 to 1.0)",
                    },
                },
                "required": ["category", "content"],
            },
            func=self._store_memory,
        )

    async def _store_memory(self, category: str, content: str, context: str = None, confidence: float = 1.0) -> str:
        """
        Store a memory entry.

        Args:
            category: The memory category
            content: The memory content
            context: Optional context
            confidence: Confidence level

        Returns:
            Confirmation message
        """
        try:
            await self.base_memory.store_memory(
                category=category,
                content=content,
                context=context,
                confidence=confidence,
            )

            return f"Stored {category} memory: {content}"

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return f"Failed to store memory: {str(e)}"


class AgentBaseMemorySearchTool(LLMTool):
    """
    Tool for searching memories in agent base memory.

    This tool allows the LLM to actively search for relevant information
    from its long-term memory when needed for current conversations.
    """

    def __init__(self, base_memory: AgentBaseMemory):
        """
        Initialize the memory search tool.

        Args:
            base_memory: The agent base memory instance to use
        """
        self.base_memory = base_memory

        super().__init__(
            name="search_memories",
            description="""Search my long-term memory for information relevant to the current conversation.
            Use this tool when you need to recall:
            - Previous facts you've learned
            - Patterns you've observed
            - User preferences you've discovered
            - Your own capabilities or limitations

            The search looks through all your stored memories to find relevant information.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories. Use keywords related to what you're looking for.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "Maximum number of memories to retrieve",
                    },
                },
                "required": ["query"],
            },
            func=self._search_memories,
        )

    async def _search_memories(self, query: str, limit: int = 10) -> str:
        """
        Search for memories matching the query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            Formatted search results
        """
        try:
            memories = await self.base_memory.search_memories(query, limit)

            if not memories:
                return f"No memories found for query: {query}"

            # Format the results
            result_parts = [f"Found {len(memories)} memories for '{query}':"]

            for memory in memories:
                result_parts.append(f"- [{memory.category}] {memory.content}")
                if memory.context:
                    result_parts.append(f"  Context: {memory.context}")

            return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return f"Failed to search memories: {str(e)}"


class AgentBaseMemoryListTool(LLMTool):
    """
    Tool for listing memories by category in agent base memory.

    This tool allows the LLM to browse its memories organized by category,
    useful for understanding what it has learned in different areas.
    """

    def __init__(self, base_memory: AgentBaseMemory):
        """
        Initialize the memory list tool.

        Args:
            base_memory: The agent base memory instance to use
        """
        self.base_memory = base_memory

        super().__init__(
            name="list_memories",
            description="""List my memories organized by category.
            Use this tool to browse what you've learned in different areas:
            - Facts: Concrete information you've stored
            - Patterns: Behavioral patterns you've observed
            - Preferences: User or system preferences you've learned
            - Capabilities: Your own abilities or limitations you've discovered

            This helps you understand what knowledge you have available.""",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["fact", "pattern", "preference", "capability"],
                        "description": "Category of memories to list",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 20,
                        "description": "Maximum number of memories to retrieve",
                    },
                },
                "required": ["category"],
            },
            func=self._list_memories,
        )

    async def _list_memories(self, category: str, limit: int = 20) -> str:
        """
        List memories in the specified category.

        Args:
            category: The memory category to list
            limit: Maximum number of results

        Returns:
            Formatted list of memories
        """
        try:
            memories = await self.base_memory.get_memories_by_category(category, limit)

            if not memories:
                return f"No {category} memories found."

            # Format the results
            result_parts = [f"My {category} memories ({len(memories)} total):"]

            for memory in memories:
                result_parts.append(f"- {memory.content}")
                if memory.context:
                    result_parts.append(f"  Context: {memory.context}")

            return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return f"Failed to list memories: {str(e)}"


def create_base_memory_tools(base_memory: AgentBaseMemory) -> List[LLMTool]:
    """
    Create all auto-registered tools for agent base memory.

    This function creates the standard set of tools that are automatically
    registered when an agent enables base memory functionality.

    Args:
        base_memory: The agent base memory instance

    Returns:
        List of memory tools to register with the agent
    """
    tools = [
        AgentBaseMemoryStoreTool(base_memory),
        AgentBaseMemorySearchTool(base_memory),
        AgentBaseMemoryListTool(base_memory),
    ]

    logger.info(f"Created {len(tools)} base memory tools")
    return tools
