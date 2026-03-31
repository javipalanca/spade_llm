"""Agent interaction memory implementation for storing specific information about agent-to-agent interactions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..tools import LLMTool

logger = logging.getLogger("spade_llm.memory.interaction")


class AgentInteractionMemory:
    """
    Manages memory for agent-to-agent interactions.

    Stores specific, useful information that an agent learns about other agents
    during interactions, such as APIs, preferences, capabilities, etc.
    """

    def __init__(self, agent_id: str, memory_path: Optional[str] = None):
        """
        Initialize agent interaction memory.

        Args:
            agent_id: The JID of the agent owning this memory
            memory_path: Optional custom memory storage path. If None, uses
                        default path "spade_llm/data/agent_memory".
        """
        self.agent_id = agent_id

        # Set up storage directory and file path
        if memory_path is None:
            # Use default path
            base_dir = Path("spade_llm/data/agent_memory")
        else:
            base_dir = Path(memory_path)

        # Create directory if it doesn't exist
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from agent_id (sanitize for filesystem)
        safe_agent_id = agent_id.replace("@", "_").replace("/", "_")
        self.storage_path = base_dir / f"{safe_agent_id}_interactions.json"

        # Load existing interactions
        self.interactions = self._load_interactions()

        logger.info(f"Initialized AgentInteractionMemory for {agent_id} at {self.storage_path}")

    def _load_interactions(self) -> Dict:
        """Load interactions from JSON file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not load interactions from {self.storage_path}: {e}")
                return {"agent_id": self.agent_id, "interactions": {}}

        return {"agent_id": self.agent_id, "interactions": {}}

    def _save_interactions(self):
        """Save interactions to JSON file."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.interactions, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved interactions to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving interactions to {self.storage_path}: {e}")

    def add_information(self, conversation_id: str, information: str) -> str:
        """
        Add specific information about an agent interaction.

        Args:
            conversation_id: The conversation ID (usually "agent1@domain_agent2@domain")
            information: Specific useful information to remember

        Returns:
            Confirmation message
        """
        # Initialize conversation entry if needed
        if conversation_id not in self.interactions["interactions"]:
            self.interactions["interactions"][conversation_id] = []

        # Add information with timestamp
        info_entry = {"content": information, "timestamp": datetime.now().isoformat()}

        self.interactions["interactions"][conversation_id].append(info_entry)

        # Save to file
        self._save_interactions()

        logger.info(f"Added interaction memory for {conversation_id}: {information}")
        return f"Remembered information about this interaction: {information}"

    def get_information(self, conversation_id: str) -> List[str]:
        """
        Get stored information for a specific conversation.

        Args:
            conversation_id: The conversation ID to retrieve info for

        Returns:
            List of information strings for this interaction
        """
        if conversation_id not in self.interactions["interactions"]:
            return []

        # Extract just the content from stored entries
        return [entry["content"] for entry in self.interactions["interactions"][conversation_id]]

    def get_all_interactions(self) -> Dict:
        """Get all stored interactions."""
        return self.interactions["interactions"]

    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear memory for a specific conversation.

        Args:
            conversation_id: The conversation ID to clear

        Returns:
            True if conversation was found and cleared, False otherwise
        """
        if conversation_id in self.interactions["interactions"]:
            del self.interactions["interactions"][conversation_id]
            self._save_interactions()
            logger.info(f"Cleared interaction memory for {conversation_id}")
            return True
        return False

    def get_context_summary(self, conversation_id: str) -> Optional[str]:
        """
        Get a formatted summary of stored information for context injection.

        Args:
            conversation_id: The conversation ID

        Returns:
            Formatted summary string or None if no information stored
        """
        info_list = self.get_information(conversation_id)

        if not info_list:
            return None

        # Format as a concise summary for context
        summary = "Previous interaction notes:\n"
        for info in info_list:
            summary += f"- {info}\n"

        return summary.strip()


class AgentMemoryTool(LLMTool):
    """
    Tool for agents to remember specific information about their interactions with other agents.
    """

    def __init__(self, interaction_memory: AgentInteractionMemory):
        """
        Initialize the agent memory tool.

        Args:
            interaction_memory: The interaction memory manager to use
        """
        self.interaction_memory = interaction_memory

        super().__init__(
            name="remember_interaction_info",
            description="""Remember specific, useful information about THIS interaction with another agent.
            Use this to store concrete details like APIs, authentication methods, data formats,
            preferences, capabilities, or technical specifications that will be helpful in future
            interactions with the same agent. Be very selective - only store truly useful
            information that improves future collaboration. Do NOT store general conversation
            content or pleasantries.""",
            parameters={
                "type": "object",
                "properties": {
                    "information": {
                        "type": "string",
                        "description": "Specific useful information to remember about this agent interaction (APIs, preferences, capabilities, etc.)",
                    }
                },
                "required": ["information"],
            },
            func=self._remember_information,
        )

    async def _remember_information(self, information: str) -> str:
        """
        Store information about the current interaction.

        Note: The conversation_id will need to be injected by the calling context.
        For now, we'll use a placeholder that should be replaced by the actual implementation.

        Args:
            information: The information to remember

        Returns:
            Confirmation message
        """
        # TODO: This needs to be called with conversation context
        # For now, we'll store with a placeholder that needs to be handled
        # by the integration with LLMAgent/LLMBehaviour

        # This is a placeholder - the actual conversation_id should be injected
        # by the calling context in LLMBehaviour
        conversation_id = getattr(self, "_current_conversation_id", "unknown_conversation")

        return self.interaction_memory.add_information(conversation_id, information)

    def set_conversation_id(self, conversation_id: str):
        """
        Set the current conversation ID for this tool execution.

        This should be called by the LLMBehaviour before tool execution.

        Args:
            conversation_id: The current conversation ID
        """
        self._current_conversation_id = conversation_id
