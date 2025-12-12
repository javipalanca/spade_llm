"""Context management for LLM conversations."""

import logging
from typing import Any, Dict, List, Optional, Set

from spade.message import Message

from ._types import (
    ContextMessage,
    create_assistant_message,
    create_system_message,
    create_tool_result_message,
    spade_message_to_user_message,
)
from .management import ContextManagement, NoContextManagement

logger = logging.getLogger("spade_llm.context")


class ContextManager:
    """
    Manages the conversation context for LLM interactions.

    Features:
    - Stores conversation history
    - Manages token limits
    - Maintains system prompts and user objectives
    - Handles persistent storage between behaviour activations
    - Supports conversation-specific contexts
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        context_management: Optional[ContextManagement] = None,
    ):
        """
        Initialize the context manager.

        Args:
            max_tokens: Maximum number of tokens to maintain in context
            system_prompt: Optional system instructions for the LLM
            context_management: Optional context management strategy
        """
        self.max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._values = {}

        # Store messages by conversation ID
        self._conversations: Dict[str, List[ContextMessage]] = {}
        # Store conversation metadata (sender_id, receiver_id) by conversation ID
        self._conversation_metadata: Dict[str, Dict[str, Optional[str]]] = {}
        # Set to track which conversation IDs are currently active
        self._active_conversations: Set[str] = set()
        # Current conversation ID (used when not explicitly specified)
        self._current_conversation_id: Optional[str] = None

        # Context management strategy
        self.context_management = context_management or NoContextManagement()

    def add_message_dict(
        self, message_dict: ContextMessage, conversation_id: str
    ) -> None:
        """
        Add a message from a dictionary format (useful for testing and direct API usage).

        Args:
            message_dict: Dictionary with 'role' and 'content' keys (and optionally 'tool_calls')
            conversation_id: ID of the conversation
        """
        self._current_conversation_id = conversation_id
        self._active_conversations.add(conversation_id)

        # Initialize conversation if needed
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        # Add message to the conversation
        self._conversations[conversation_id].append(message_dict)

    def add_message(self, message: Message, conversation_id: str) -> None:
        """
        Add a message to the context.

        Args:
            message: The SPADE message to add to the context
            conversation_id: ID of the conversation
        """
        self._current_conversation_id = conversation_id

        # Mark conversation as active
        self._active_conversations.add(conversation_id)

        # Initialize conversation if needed
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
            # Initialize metadata for this conversation
            self._conversation_metadata[conversation_id] = {
                "sender_id": str(message.sender),
                "receiver_id": str(message.to)
            }

        # Convert SPADE message to a format suitable for LLM context using our helper function
        user_message = spade_message_to_user_message(message)

        # Store original SPADE metadata as additional fields (these won't be sent to the LLM)
        # but are useful for debugging and advanced processing
        user_message_with_metadata = dict(user_message)
        user_message_with_metadata["sender"] = str(message.sender)
        user_message_with_metadata["receiver"] = str(message.to)
        user_message_with_metadata["thread"] = message.thread

        # Add message to the conversation
        self._conversations[conversation_id].append(user_message_with_metadata)

        # TODO : Token counting and context windowing will be implemented later

    def get_prompt(self, conversation_id: Optional[str] = None) -> List[ContextMessage]:
        """
        Get the current prompt including history formatted for LLM providers.

        Args:
            conversation_id: Optional ID of the conversation. If not provided,
                          uses the current conversation.

        Returns:
            A list of message dictionaries formatted for LLM API consumption
        """
        # Determine which conversation to use
        conv_id = conversation_id or self._current_conversation_id

        # If no conversation is specified or found, return just the system prompt
        if not conv_id or conv_id not in self._conversations:
            if self._system_prompt:
                return [create_system_message(self._system_prompt)]
            return []

        prompt = []

        # Add system prompt if available
        if self._system_prompt:
            prompt.append(create_system_message(self._system_prompt))

        # Get conversation messages
        conversation_messages = self._conversations[conv_id]

        # Apply context management strategy
        managed_messages = self.context_management.apply_context_strategy(
            conversation_messages, self._system_prompt
        )

        # Clean and add messages to prompt
        for msg in managed_messages:
            clean_message = self._clean_message_for_llm(msg)
            prompt.append(clean_message)

        return prompt

    def add_assistant_message(
        self, content: str, conversation_id: Optional[str] = None
    ) -> None:
        """
        Add an assistant (LLM) response to the context.

        Args:
            content: The assistant's response content
            conversation_id: ID of the conversation
        """
        # Determine which conversation to use
        conv_id = conversation_id or self._current_conversation_id

        if not conv_id:
            logger.warning(
                "No conversation ID provided and no current conversation set"
            )
            return

        # Initialize conversation if needed
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []

        # Add assistant response to the conversation using our helper function
        assistant_message = create_assistant_message(content)

        self._conversations[conv_id].append(assistant_message)
        logger.debug(
            f"Added assistant response to conversation {conv_id}: {content[:100]}..."
        )

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        """
        Add a tool execution result to the context.

        Args:
            tool_name: The name of the tool that was executed
            result: The result returned by the tool
            tool_call_id: The ID of the tool call this result is for
            conversation_id: Optional ID of the conversation to add the result to
        """
        # Determine which conversation to use
        conv_id = conversation_id or self._current_conversation_id

        # If no conversation is found, log a warning and return
        if not conv_id or conv_id not in self._conversations:
            logger.warning(
                f"No active conversation found to add tool result for: {tool_name}"
            )
            return

        # Use our helper function to create the tool result message
        tool_result = create_tool_result_message(result, tool_call_id)

        # Add tool name as metadata (won't be sent to LLM but useful for debugging)
        tool_result_with_metadata = dict(tool_result)
        tool_result_with_metadata["tool_name"] = tool_name

        self._conversations[conv_id].append(tool_result_with_metadata)

    def clear(self, conversation_id: Optional[str] = None) -> None:
        """
        Clear messages from the context but retain system prompt.

        Args:
            conversation_id: Optional ID of the conversation to clear.
                          If None, clears the current conversation.
                          If 'all', clears all conversations.
        """
        if conversation_id == "all":
            self._conversations = {}
            self._active_conversations = set()
            self._current_conversation_id = None
        else:
            conv_id = conversation_id or self._current_conversation_id
            if conv_id in self._conversations:
                self._conversations[conv_id] = []
                if conv_id in self._active_conversations:
                    self._active_conversations.remove(conv_id)

                # Reset current conversation if it was cleared
                if self._current_conversation_id == conv_id:
                    self._current_conversation_id = None

    def get_active_conversations(self) -> List[str]:
        """
        Get a list of all active conversation IDs.

        Returns:
            List of conversation IDs
        """
        return list(self._active_conversations)

    def set_current_conversation(self, conversation_id: str) -> bool:
        """
        Set the current conversation context.

        Args:
            conversation_id: The ID of the conversation to set as current

        Returns:
            bool: True if successful, False if conversation doesn't exist
        """
        if conversation_id in self._conversations:
            self._current_conversation_id = conversation_id
            return True
        return False

    def get_conversation_history(
        self, conversation_id: Optional[str] = None
    ) -> List[ContextMessage]:
        """
        Get the raw conversation history for a specific conversation.

        Args:
            conversation_id: Optional ID of the conversation. If not provided,
                          uses the current conversation.

        Returns:
            List of message dictionaries in the conversation
        """
        conv_id = conversation_id or self._current_conversation_id

        if not conv_id or conv_id not in self._conversations:
            return []

        return self._conversations[conv_id]

    def _clean_message_for_llm(self, msg: ContextMessage) -> ContextMessage:
        """
        Remove internal metadata from message for LLM consumption.

        Args:
            msg: Message with potential internal metadata

        Returns:
            Clean message suitable for LLM API
        """
        message_entry = {"role": msg["role"], "content": msg["content"]}

        # Include tool_call_id for tool messages
        if msg["role"] == "tool" and "tool_call_id" in msg:
            message_entry["tool_call_id"] = msg["tool_call_id"]

        # Include tool_calls if present (for assistant messages with tool calls)
        if msg.get("tool_calls"):
            message_entry["tool_calls"] = msg["tool_calls"]

        # Include name if present for user messages
        if msg["role"] == "user" and "name" in msg:
            message_entry["name"] = msg["name"]

        return message_entry

    def get_context_stats(
        self, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get context management statistics for a conversation.

        Args:
            conversation_id: Optional ID of the conversation. If not provided,
                          uses the current conversation.

        Returns:
            Dictionary with context management statistics
        """
        conv_id = conversation_id or self._current_conversation_id

        if not conv_id or conv_id not in self._conversations:
            return {}

        total_messages = len(self._conversations[conv_id])
        return self.context_management.get_stats(total_messages)

    def update_context_management(
        self, new_context_management: ContextManagement
    ) -> None:
        """
        Update the context management strategy.

        Args:
            new_context_management: New context management strategy to use
        """
        self.context_management = new_context_management

    def get_tracing_metadata(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for tracing/logging purposes.

        Args:
            conversation_id: Optional ID of the conversation. If not provided,
                          uses the current conversation.

        Returns:
            Dictionary with conversation metadata (conversation_id, sender_id, receiver_id)
        """
        conv_id = conversation_id or self._current_conversation_id
        
        metadata = {"conversation_id": conv_id}
        
        if conv_id and conv_id in self._conversation_metadata:
            metadata.update(self._conversation_metadata[conv_id])
        
        return metadata
