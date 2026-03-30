"""Context management strategies for controlling conversation history."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ._types import ContextMessage


class ContextManagement(ABC):
    """Abstract base class for context management strategies."""

    @abstractmethod
    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """
        Apply context management strategy to messages.

        Args:
            messages: List of conversation messages to manage
            system_prompt: Optional system prompt (for context)

        Returns:
            List of messages after applying the strategy
        """
        pass

    @abstractmethod
    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """
        Get statistics about context management.

        Args:
            total_messages: Total number of messages before management

        Returns:
            Dictionary with statistics about the strategy application
        """
        pass


class NoContextManagement(ContextManagement):
    """Default context management strategy that keeps all messages."""

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Return all messages unchanged."""
        return messages

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for no management strategy."""
        return {
            "strategy": "none",
            "total_messages": total_messages,
            "messages_in_context": total_messages,
            "messages_dropped": 0,
        }


class WindowSizeContext(ContextManagement):
    """Window-based context management that keeps the last N messages."""

    def __init__(self, max_messages: int = 20):
        """
        Initialize window size context management.

        Args:
            max_messages: Maximum number of messages to keep in context
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")

        self.max_messages = max_messages

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Keep only the last max_messages from the conversation."""
        if len(messages) <= self.max_messages:
            return messages

        return messages[-self.max_messages :]

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for window size strategy."""
        messages_in_context = min(total_messages, self.max_messages)
        messages_dropped = max(0, total_messages - self.max_messages)

        return {
            "strategy": "window_size",
            "max_messages": self.max_messages,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
        }


class SmartWindowSizeContext(ContextManagement):
    """Smart window-based context management with optional initial message preservation and tool prioritization."""

    def __init__(
        self,
        max_messages: int = 20,
        preserve_initial: int = 0,
        prioritize_tools: bool = False,
    ):
        """
        Initialize smart window size context management.

        Args:
            max_messages: Maximum number of messages to keep in context
            preserve_initial: Number of initial messages to always preserve (0 = disabled)
            prioritize_tools: Whether to prioritize tool results in selection
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        if preserve_initial < 0:
            raise ValueError("preserve_initial must be >= 0")
        if preserve_initial >= max_messages:
            raise ValueError("preserve_initial must be less than max_messages")

        self.max_messages = max_messages
        self.preserve_initial = preserve_initial
        self.prioritize_tools = prioritize_tools

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Apply smart context management strategy."""
        if len(messages) <= self.max_messages:
            return messages

        if self.preserve_initial == 0 and not self.prioritize_tools:
            return self._sliding_window_with_pairs(messages)

        if self.preserve_initial > 0 and not self.prioritize_tools:
            return self._preserve_initial_only(messages)

        if self.preserve_initial == 0 and self.prioritize_tools:
            return self._prioritize_tools_only(messages)

        return self._smart_combination(messages)

    def _sliding_window_with_pairs(self, messages: List[ContextMessage]) -> List[ContextMessage]:
        """Simple sliding window that preserves tool pairs."""
        tool_pairs = self._find_tool_pairs(messages)
        selected_indices = set()
        current_pos = len(messages) - 1

        while len(selected_indices) < self.max_messages and current_pos >= 0:
            # Check if this message is part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if current_pos in pair_indices:
                    pair_for_msg = pair_indices
                    break

            if pair_for_msg:
                # Add all messages of the pair if we have space
                if len(selected_indices) + len(pair_for_msg) <= self.max_messages:
                    selected_indices.update(pair_for_msg)
                    current_pos = min(pair_for_msg) - 1
                else:
                    # Not enough space for the pair, skip it
                    current_pos = min(pair_for_msg) - 1
            else:
                # Regular message, add it
                selected_indices.add(current_pos)
                current_pos -= 1

        # Return messages in original order
        return [messages[i] for i in sorted(selected_indices)]

    def _preserve_initial_only(self, messages: List[ContextMessage]) -> List[ContextMessage]:
        """Preserve initial N messages plus recent messages to fill window, respecting tool pairs."""
        initial = messages[: self.preserve_initial]
        remaining_space = self.max_messages - len(initial)

        if remaining_space <= 0:
            return initial[: self.max_messages]

        if len(messages) <= self.preserve_initial + remaining_space:
            return messages

        # Find tool pairs
        tool_pairs = self._find_tool_pairs(messages)
        selected_indices = set(range(len(initial)))

        # Fill remaining space from the end, respecting tool pairs
        current_pos = len(messages) - 1
        while remaining_space > 0 and current_pos >= self.preserve_initial:
            if current_pos in selected_indices:
                current_pos -= 1
                continue

            # Check if this message is part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if current_pos in pair_indices:
                    pair_for_msg = pair_indices
                    break

            if pair_for_msg:
                # Only add if all messages in pair fit and are after initial messages
                pair_after_initial = [idx for idx in pair_for_msg if idx >= self.preserve_initial]
                if (
                    len(pair_after_initial) == len(pair_for_msg)
                    and remaining_space >= len(pair_for_msg)
                    and not any(idx in selected_indices for idx in pair_for_msg)
                ):
                    selected_indices.update(pair_for_msg)
                    remaining_space -= len(pair_for_msg)
                current_pos = min(pair_for_msg) - 1
            else:
                selected_indices.add(current_pos)
                remaining_space -= 1
                current_pos -= 1

        return [messages[i] for i in sorted(selected_indices)]

    def _find_tool_pairs(self, messages: List[ContextMessage]) -> List[tuple]:
        """Find assistant-tool message pairs, handling multiple tool calls and results."""
        pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            # Look for assistant messages with tool_calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls", [])
                call_ids = {call.get("id") for call in tool_calls if call.get("id")}

                # Find all corresponding tool result messages
                tool_result_indices = []
                j = i + 1

                # Look for tool results that match the call IDs
                while j < len(messages) and len(call_ids) > 0:
                    if messages[j].get("role") == "tool":
                        tool_call_id = messages[j].get("tool_call_id")
                        if tool_call_id in call_ids:
                            tool_result_indices.append(j)
                            call_ids.remove(tool_call_id)
                    elif messages[j].get("role") in ["user", "assistant"]:
                        # Stop looking if we hit another conversation turn
                        break
                    j += 1

                # Only create pairs if we found all tool results
                if len(call_ids) == 0 and tool_result_indices:
                    # Create a group that includes the assistant message and all its tool results
                    pair_indices = [i] + tool_result_indices
                    pairs.append(tuple(pair_indices))
                    i = max(tool_result_indices) + 1
                else:
                    i += 1
            else:
                i += 1
        return pairs

    def _prioritize_tools_only(self, messages: List[ContextMessage]) -> List[ContextMessage]:
        """Prioritize tool results while preserving tool call/result pairs."""
        tool_pairs = self._find_tool_pairs(messages)
        selected_indices = set()

        # First, add all tool pairs that fit (from most recent)
        for pair_indices in reversed(tool_pairs):
            if len(selected_indices) + len(pair_indices) <= self.max_messages:
                selected_indices.update(pair_indices)

        # Fill remaining space with non-tool messages
        remaining_space = self.max_messages - len(selected_indices)
        current_pos = len(messages) - 1

        while remaining_space > 0 and current_pos >= 0:
            # Skip messages that are part of tool pairs already selected
            is_in_pair = any(current_pos in pair_indices for pair_indices in tool_pairs)
            if current_pos not in selected_indices and not is_in_pair:
                selected_indices.add(current_pos)
                remaining_space -= 1
            current_pos -= 1

        return [messages[i] for i in sorted(selected_indices)]

    def _smart_combination(self, messages: List[ContextMessage]) -> List[ContextMessage]:
        """Combine initial preservation with tool prioritization while preserving tool pairs."""
        initial = messages[: self.preserve_initial]
        available_space = self.max_messages - len(initial)

        if available_space <= 0:
            return initial[: self.max_messages]

        # Find tool pairs in the entire conversation
        tool_pairs = self._find_tool_pairs(messages)

        # Filter pairs to only those after initial messages
        relevant_pairs = [
            pair_indices for pair_indices in tool_pairs if all(idx >= self.preserve_initial for idx in pair_indices)
        ]

        selected_indices = set(range(len(initial)))

        # Add tool pairs first (from most recent)
        for pair_indices in reversed(relevant_pairs):
            if available_space >= len(pair_indices):
                selected_indices.update(pair_indices)
                available_space -= len(pair_indices)

        # Fill remaining space with other messages from the end
        current_pos = len(messages) - 1
        while available_space > 0 and current_pos >= self.preserve_initial:
            # Skip messages that are part of tool pairs or already selected
            is_in_pair = any(current_pos in pair_indices for pair_indices in relevant_pairs)
            if current_pos not in selected_indices and not is_in_pair:
                selected_indices.add(current_pos)
                available_space -= 1
            current_pos -= 1

        return [messages[i] for i in sorted(selected_indices)]

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for smart window size strategy."""
        messages_in_context = min(total_messages, self.max_messages)
        messages_dropped = max(0, total_messages - self.max_messages)

        return {
            "strategy": "smart_window_size",
            "max_messages": self.max_messages,
            "preserve_initial": self.preserve_initial,
            "prioritize_tools": self.prioritize_tools,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
        }
