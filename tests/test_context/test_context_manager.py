"""Tests for ContextManager class."""

import pytest

from spade_llm.context import ContextManager
from spade_llm.context._types import create_assistant_message, create_user_message
from spade_llm.context.management import NoContextManagement, SmartWindowSizeContext, WindowSizeContext


class TestContextManagerInitialization:
    """Test ContextManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        cm = ContextManager()

        assert cm.max_tokens == 4096
        assert cm._system_prompt is None
        assert cm._conversations == {}
        assert cm._active_conversations == set()
        assert cm._current_conversation_id is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        cm = ContextManager(max_tokens=8192, system_prompt="Custom system prompt")

        assert cm.max_tokens == 8192
        assert cm._system_prompt == "Custom system prompt"
        assert cm._conversations == {}
        assert cm._active_conversations == set()

    def test_init_values_dict(self):
        """Test that _values dict is initialized."""
        cm = ContextManager()
        assert hasattr(cm, "_values")
        assert isinstance(cm._values, dict)


class TestMessageHandling:
    """Test message handling functionality."""

    def test_add_message_dict_new_conversation(self, context_manager, conversation_id):
        """Test adding a message dict to a new conversation."""
        message_dict = create_user_message("Test message content")

        context_manager.add_message_dict(message_dict, conversation_id)

        assert conversation_id in context_manager._conversations
        assert conversation_id in context_manager._active_conversations
        assert context_manager._current_conversation_id == conversation_id
        assert len(context_manager._conversations[conversation_id]) == 1
        assert context_manager._conversations[conversation_id][0] == message_dict

    def test_add_message_dict_existing_conversation(self, context_manager, conversation_id):
        """Test adding a message dict to an existing conversation."""
        # Add first message
        msg1 = create_user_message("First message")
        context_manager.add_message_dict(msg1, conversation_id)

        # Add second message
        msg2 = create_user_message("Second message")
        context_manager.add_message_dict(msg2, conversation_id)

        messages = context_manager._conversations[conversation_id]
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2

    def test_add_spade_message(self, context_manager, mock_message, conversation_id):
        """Test adding a SPADE message to context."""
        context_manager.add_message(mock_message, conversation_id)

        assert conversation_id in context_manager._conversations
        assert conversation_id in context_manager._active_conversations
        assert context_manager._current_conversation_id == conversation_id

        messages = context_manager._conversations[conversation_id]
        assert len(messages) == 1

        # Check that message was converted properly
        stored_msg = messages[0]
        assert stored_msg["role"] == "user"
        assert stored_msg["content"] == mock_message.body
        assert stored_msg["sender"] == str(mock_message.sender)
        assert stored_msg["receiver"] == str(mock_message.to)
        assert stored_msg["thread"] == mock_message.thread

    def test_add_assistant_message(self, context_manager, conversation_id):
        """Test adding an assistant message."""
        content = "Assistant response"

        # First add a user message to establish the conversation
        user_msg = create_user_message("User message")
        context_manager.add_message_dict(user_msg, conversation_id)

        # Add assistant message
        context_manager.add_assistant_message(content, conversation_id)

        messages = context_manager._conversations[conversation_id]
        assert len(messages) == 2

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == content

    def test_add_assistant_message_no_conversation(self, context_manager):
        """Test adding assistant message without active conversation."""
        # This should log a warning but not crash
        context_manager.add_assistant_message("Test content")

        # Should not create any conversation
        assert len(context_manager._conversations) == 0

    def test_add_tool_result(self, context_manager, conversation_id):
        """Test adding a tool result to context."""
        # First establish a conversation
        user_msg = create_user_message("User message")
        context_manager.add_message_dict(user_msg, conversation_id)

        # Add tool result
        result = {"status": "success", "data": "tool output"}
        tool_call_id = "call_123"

        context_manager.add_tool_result("test_tool", result, tool_call_id, conversation_id)

        messages = context_manager._conversations[conversation_id]
        assert len(messages) == 2

        tool_msg = messages[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == str(result)
        assert tool_msg["tool_call_id"] == tool_call_id
        assert tool_msg["tool_name"] == "test_tool"  # metadata

    def test_add_tool_result_no_conversation(self, context_manager):
        """Test adding tool result without active conversation logs warning."""
        # This should log a warning but not crash
        context_manager.add_tool_result("test_tool", {"result": "test"}, "call_123")

        # Should not create any conversation
        assert len(context_manager._conversations) == 0


class TestPromptGeneration:
    """Test prompt generation functionality."""

    def test_get_prompt_empty_no_system(self, context_manager_no_system):
        """Test getting prompt with no conversations and no system prompt."""
        prompt = context_manager_no_system.get_prompt()
        assert prompt == []

    def test_get_prompt_empty_with_system(self, context_manager):
        """Test getting prompt with no conversations but with system prompt."""
        prompt = context_manager.get_prompt()

        assert len(prompt) == 1
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"] == "You are a helpful test assistant."

    def test_get_prompt_with_messages(self, context_manager, conversation_id):
        """Test getting prompt with messages."""
        # Add messages to conversation
        user_msg = create_user_message("Hello")
        assistant_msg = create_assistant_message("Hi there!")

        context_manager.add_message_dict(user_msg, conversation_id)
        context_manager.add_message_dict(assistant_msg, conversation_id)

        prompt = context_manager.get_prompt(conversation_id)

        # Should have system + user + assistant
        assert len(prompt) == 3
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "Hello"
        assert prompt[2]["role"] == "assistant"
        assert prompt[2]["content"] == "Hi there!"

    def test_get_prompt_filters_metadata(self, context_manager, mock_message, conversation_id):
        """Test that prompt generation filters out internal metadata."""
        context_manager.add_message(mock_message, conversation_id)

        prompt = context_manager.get_prompt(conversation_id)

        # Should have system + user message
        assert len(prompt) == 2
        user_message = prompt[1]

        # Should only have role and content, not sender/receiver/thread
        assert set(user_message.keys()) == {"role", "content", "name"}
        assert "sender" not in user_message
        assert "receiver" not in user_message
        assert "thread" not in user_message

    def test_get_prompt_includes_tool_calls(self, context_manager, conversation_id):
        """Test that tool calls are included in prompt."""
        # Create assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
                }
            ],
        }

        context_manager.add_message_dict(assistant_msg, conversation_id)

        prompt = context_manager.get_prompt(conversation_id)

        # Should include tool_calls in the prompt
        assert len(prompt) == 2  # system + assistant
        assistant_prompt = prompt[1]
        assert "tool_calls" in assistant_prompt
        assert assistant_prompt["tool_calls"] == assistant_msg["tool_calls"]

    def test_get_prompt_includes_tool_call_id(self, context_manager, conversation_id):
        """Test that tool messages include tool_call_id."""
        # Add a tool result message
        tool_msg = {
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call_123",
            "tool_name": "test_tool",  # This is metadata and should not appear in prompt
        }

        context_manager.add_message_dict(tool_msg, conversation_id)

        prompt = context_manager.get_prompt(conversation_id)

        # Should have system + tool message
        assert len(prompt) == 2
        tool_prompt = prompt[1]
        assert tool_prompt["role"] == "tool"
        assert tool_prompt["content"] == "Tool result"
        assert tool_prompt["tool_call_id"] == "call_123"
        assert "tool_name" not in tool_prompt  # Metadata should be filtered out


class TestConversationManagement:
    """Test conversation management functionality."""

    def test_conversation_isolation(self, context_manager, conversation_id, different_conversation_id):
        """Test that different conversations are isolated."""
        # Add messages to different conversations
        msg1 = create_user_message("Message in conversation 1")
        msg2 = create_user_message("Message in conversation 2")

        context_manager.add_message_dict(msg1, conversation_id)
        context_manager.add_message_dict(msg2, different_conversation_id)

        # Check isolation
        conv1_messages = context_manager._conversations[conversation_id]
        conv2_messages = context_manager._conversations[different_conversation_id]

        assert len(conv1_messages) == 1
        assert len(conv2_messages) == 1
        assert conv1_messages[0] != conv2_messages[0]

        # Check prompts are different
        prompt1 = context_manager.get_prompt(conversation_id)
        prompt2 = context_manager.get_prompt(different_conversation_id)

        assert prompt1[1]["content"] == "Message in conversation 1"
        assert prompt2[1]["content"] == "Message in conversation 2"

    def test_get_active_conversations(self, context_manager, conversation_id, different_conversation_id):
        """Test getting list of active conversations."""
        # Initially no active conversations
        assert context_manager.get_active_conversations() == []

        # Add messages to create active conversations
        msg1 = create_user_message("Message 1")
        msg2 = create_user_message("Message 2")

        context_manager.add_message_dict(msg1, conversation_id)
        context_manager.add_message_dict(msg2, different_conversation_id)

        active = context_manager.get_active_conversations()
        assert set(active) == {conversation_id, different_conversation_id}

    def test_set_current_conversation(self, context_manager, conversation_id):
        """Test setting current conversation."""
        # Create a conversation first
        msg = create_user_message("Test message")
        context_manager.add_message_dict(msg, conversation_id)

        # Set it as current
        result = context_manager.set_current_conversation(conversation_id)
        assert result is True
        assert context_manager._current_conversation_id == conversation_id

    def test_set_current_conversation_nonexistent(self, context_manager):
        """Test setting current conversation to non-existent ID."""
        result = context_manager.set_current_conversation("nonexistent")
        assert result is False
        assert context_manager._current_conversation_id is None

    def test_get_conversation_history(self, context_manager, conversation_id):
        """Test getting conversation history."""
        # Add some messages
        user_msg = create_user_message("User message")
        assistant_msg = create_assistant_message("Assistant message")

        context_manager.add_message_dict(user_msg, conversation_id)
        context_manager.add_message_dict(assistant_msg, conversation_id)

        history = context_manager.get_conversation_history(conversation_id)

        assert len(history) == 2
        assert history[0] == user_msg
        assert history[1] == assistant_msg

    def test_get_conversation_history_nonexistent(self, context_manager):
        """Test getting history for non-existent conversation."""
        history = context_manager.get_conversation_history("nonexistent")
        assert history == []


class TestContextClearing:
    """Test context clearing functionality."""

    def test_clear_specific_conversation(self, context_manager, conversation_id, different_conversation_id):
        """Test clearing a specific conversation."""
        # Add messages to two conversations
        msg1 = create_user_message("Message 1")
        msg2 = create_user_message("Message 2")

        context_manager.add_message_dict(msg1, conversation_id)
        context_manager.add_message_dict(msg2, different_conversation_id)

        # Clear first conversation
        context_manager.clear(conversation_id)

        # First conversation should be cleared, second should remain
        assert context_manager._conversations[conversation_id] == []
        assert len(context_manager._conversations[different_conversation_id]) == 1
        assert conversation_id not in context_manager._active_conversations
        assert different_conversation_id in context_manager._active_conversations

    def test_clear_current_conversation(self, context_manager, conversation_id):
        """Test clearing current conversation."""
        # Add message and set as current
        msg = create_user_message("Test message")
        context_manager.add_message_dict(msg, conversation_id)

        # Clear without specifying ID (should clear current)
        context_manager.clear()

        assert context_manager._conversations[conversation_id] == []
        assert conversation_id not in context_manager._active_conversations
        assert context_manager._current_conversation_id is None

    def test_clear_all_conversations(self, context_manager, conversation_id, different_conversation_id):
        """Test clearing all conversations."""
        # Add messages to multiple conversations
        msg1 = create_user_message("Message 1")
        msg2 = create_user_message("Message 2")

        context_manager.add_message_dict(msg1, conversation_id)
        context_manager.add_message_dict(msg2, different_conversation_id)

        # Clear all
        context_manager.clear("all")

        assert context_manager._conversations == {}
        assert context_manager._active_conversations == set()
        assert context_manager._current_conversation_id is None

    def test_clear_nonexistent_conversation(self, context_manager):
        """Test clearing a non-existent conversation doesn't error."""
        # This should not raise an error
        context_manager.clear("nonexistent")

        # State should remain unchanged
        assert context_manager._conversations == {}
        assert context_manager._active_conversations == set()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_prompt_with_current_conversation_none(self, context_manager):
        """Test getting prompt when current conversation is None."""
        # Should return just system prompt
        prompt = context_manager.get_prompt()
        assert len(prompt) <= 1  # 0 if no system prompt, 1 if system prompt exists

    def test_multiple_assistant_messages(self, context_manager, conversation_id):
        """Test adding multiple assistant messages in sequence."""
        context_manager.add_assistant_message("Response 1", conversation_id)
        context_manager.add_assistant_message("Response 2", conversation_id)

        messages = context_manager._conversations[conversation_id]
        assert len(messages) == 2
        assert all(msg["role"] == "assistant" for msg in messages)

    def test_empty_message_content(self, context_manager, conversation_id):
        """Test handling empty message content."""
        empty_msg = create_user_message("")
        context_manager.add_message_dict(empty_msg, conversation_id)

        prompt = context_manager.get_prompt(conversation_id)
        user_message = prompt[1]
        assert user_message["content"] == ""

    def test_none_message_content(self, context_manager, conversation_id):
        """Test handling None message content."""
        none_msg = {"role": "assistant", "content": None}
        context_manager.add_message_dict(none_msg, conversation_id)

        prompt = context_manager.get_prompt(conversation_id)
        assistant_message = prompt[1]
        assert assistant_message["content"] is None


class TestNoContextManagement:
    """Test NoContextManagement strategy."""

    def test_init_no_context_management(self):
        """Test NoContextManagement initialization."""
        strategy = NoContextManagement()
        assert strategy is not None

    def test_apply_context_strategy_returns_all_messages(self):
        """Test that NoContextManagement returns all messages unchanged."""
        strategy = NoContextManagement()
        messages = [
            create_user_message("Message 1"),
            create_assistant_message("Response 1"),
            create_user_message("Message 2"),
            create_assistant_message("Response 2"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert result == messages
        assert len(result) == 4

    def test_apply_context_strategy_with_system_prompt(self):
        """Test NoContextManagement with system prompt (should ignore it)."""
        strategy = NoContextManagement()
        messages = [create_user_message("Test")]

        result = strategy.apply_context_strategy(messages, "System prompt")
        assert result == messages

    def test_get_stats(self):
        """Test NoContextManagement statistics."""
        strategy = NoContextManagement()
        stats = strategy.get_stats(10)

        expected = {"strategy": "none", "total_messages": 10, "messages_in_context": 10, "messages_dropped": 0}
        assert stats == expected


class TestWindowSizeContext:
    """Test WindowSizeContext strategy."""

    def test_init_window_size_context(self):
        """Test WindowSizeContext initialization."""
        strategy = WindowSizeContext(max_messages=5)
        assert strategy.max_messages == 5

    def test_init_invalid_max_messages(self):
        """Test WindowSizeContext with invalid max_messages."""
        with pytest.raises(ValueError, match="max_messages must be greater than 0"):
            WindowSizeContext(max_messages=0)

        with pytest.raises(ValueError, match="max_messages must be greater than 0"):
            WindowSizeContext(max_messages=-1)

    def test_apply_context_strategy_under_limit(self):
        """Test WindowSizeContext when messages are under the limit."""
        strategy = WindowSizeContext(max_messages=5)
        messages = [
            create_user_message("Message 1"),
            create_assistant_message("Response 1"),
            create_user_message("Message 2"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert result == messages
        assert len(result) == 3

    def test_apply_context_strategy_over_limit(self):
        """Test WindowSizeContext when messages exceed the limit."""
        strategy = WindowSizeContext(max_messages=3)
        messages = [
            create_user_message("Message 1"),
            create_assistant_message("Response 1"),
            create_user_message("Message 2"),
            create_assistant_message("Response 2"),
            create_user_message("Message 3"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 3
        # Should keep the last 3 messages
        assert result == messages[-3:]

    def test_apply_context_strategy_exact_limit(self):
        """Test WindowSizeContext when messages exactly match the limit."""
        strategy = WindowSizeContext(max_messages=3)
        messages = [
            create_user_message("Message 1"),
            create_assistant_message("Response 1"),
            create_user_message("Message 2"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert result == messages
        assert len(result) == 3

    def test_get_stats_under_limit(self):
        """Test WindowSizeContext statistics when under limit."""
        strategy = WindowSizeContext(max_messages=10)
        stats = strategy.get_stats(5)

        expected = {
            "strategy": "window_size",
            "max_messages": 10,
            "total_messages": 5,
            "messages_in_context": 5,
            "messages_dropped": 0,
        }
        assert stats == expected

    def test_get_stats_over_limit(self):
        """Test WindowSizeContext statistics when over limit."""
        strategy = WindowSizeContext(max_messages=5)
        stats = strategy.get_stats(10)

        expected = {
            "strategy": "window_size",
            "max_messages": 5,
            "total_messages": 10,
            "messages_in_context": 5,
            "messages_dropped": 5,
        }
        assert stats == expected


class TestSmartWindowSizeContext:
    """Test SmartWindowSizeContext strategy."""

    def test_init_smart_window_size_context(self):
        """Test SmartWindowSizeContext initialization."""
        strategy = SmartWindowSizeContext(max_messages=10, preserve_initial=2, prioritize_tools=True)
        assert strategy.max_messages == 10
        assert strategy.preserve_initial == 2
        assert strategy.prioritize_tools is True

    def test_init_invalid_max_messages(self):
        """Test SmartWindowSizeContext with invalid max_messages."""
        with pytest.raises(ValueError, match="max_messages must be greater than 0"):
            SmartWindowSizeContext(max_messages=0)

    def test_init_invalid_preserve_initial(self):
        """Test SmartWindowSizeContext with invalid preserve_initial."""
        with pytest.raises(ValueError, match="preserve_initial must be >= 0"):
            SmartWindowSizeContext(max_messages=10, preserve_initial=-1)

    def test_init_preserve_initial_too_large(self):
        """Test SmartWindowSizeContext with preserve_initial >= max_messages."""
        with pytest.raises(ValueError, match="preserve_initial must be less than max_messages"):
            SmartWindowSizeContext(max_messages=5, preserve_initial=5)

    def test_apply_context_strategy_under_limit(self):
        """Test SmartWindowSizeContext when messages are under the limit."""
        strategy = SmartWindowSizeContext(max_messages=10)
        messages = [
            create_user_message("Message 1"),
            create_assistant_message("Response 1"),
            create_user_message("Message 2"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert result == messages
        assert len(result) == 3

    def test_sliding_window_with_pairs_basic(self):
        """Test sliding window with tool pairs preservation."""
        strategy = SmartWindowSizeContext(max_messages=4, preserve_initial=0, prioritize_tools=False)

        # Create messages with tool call and result
        messages = [
            create_user_message("Question 1"),
            {
                "role": "assistant",
                "content": "Let me search for that",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "Search result", "tool_call_id": "call_123", "tool_name": "search"},
            create_user_message("Question 2"),
            create_assistant_message("Final answer"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 4
        # Should keep the last 4 messages, preserving the tool pair
        assert result == messages[1:]

    def test_preserve_initial_only(self):
        """Test preserving initial messages only."""
        strategy = SmartWindowSizeContext(max_messages=4, preserve_initial=2, prioritize_tools=False)

        messages = [
            create_user_message("Initial 1"),
            create_assistant_message("Initial response"),
            create_user_message("Middle message"),
            create_assistant_message("Middle response"),
            create_user_message("Recent message"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 4
        # Should have first 2 + last 2 messages
        assert result[0] == messages[0]  # Initial 1
        assert result[1] == messages[1]  # Initial response
        assert result[2] == messages[3]  # Middle response
        assert result[3] == messages[4]  # Recent message

    def test_prioritize_tools_only(self):
        """Test prioritizing tools without initial preservation."""
        strategy = SmartWindowSizeContext(max_messages=4, preserve_initial=0, prioritize_tools=True)

        messages = [
            create_user_message("Question 1"),
            {
                "role": "assistant",
                "content": "Let me search",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "Search result", "tool_call_id": "call_123", "tool_name": "search"},
            create_user_message("Question 2"),
            create_assistant_message("No tools here"),
            create_user_message("Question 3"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 4
        # Should prioritize tool pair and include recent messages
        tool_pair_indices = [1, 2]  # Assistant with tool call and tool result
        result_indices = [messages.index(msg) for msg in result]
        assert all(idx in result_indices for idx in tool_pair_indices)

    def test_smart_combination(self):
        """Test combination of initial preservation and tool prioritization."""
        strategy = SmartWindowSizeContext(max_messages=6, preserve_initial=2, prioritize_tools=True)

        messages = [
            create_user_message("Initial 1"),
            create_assistant_message("Initial response"),
            create_user_message("Middle"),
            {
                "role": "assistant",
                "content": "Tool call",
                "tool_calls": [{"id": "call_456", "type": "function", "function": {"name": "tool", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Tool result", "tool_call_id": "call_456", "tool_name": "tool"},
            create_user_message("After tool"),
            create_assistant_message("Final"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 6
        # Should preserve initial 2 + prioritize tool pair + fill remaining
        assert result[0] == messages[0]  # Initial 1
        assert result[1] == messages[1]  # Initial response
        # Should include tool pair [3, 4]
        tool_pair_indices = [3, 4]
        result_indices = [messages.index(msg) for msg in result]
        assert all(idx in result_indices for idx in tool_pair_indices)

    def test_find_tool_pairs_single_tool(self):
        """Test finding tool pairs with single tool call."""
        strategy = SmartWindowSizeContext(max_messages=10)

        messages = [
            create_user_message("Question"),
            {
                "role": "assistant",
                "content": "Tool call",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "Result", "tool_call_id": "call_123", "tool_name": "search"},
            create_user_message("Follow up"),
        ]

        pairs = strategy._find_tool_pairs(messages)
        assert len(pairs) == 1
        assert pairs[0] == (1, 2)  # Assistant message + tool result

    def test_find_tool_pairs_multiple_tools(self):
        """Test finding tool pairs with multiple tool calls."""
        strategy = SmartWindowSizeContext(max_messages=10)

        messages = [
            create_user_message("Question"),
            {
                "role": "assistant",
                "content": "Multiple tools",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "calc", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "Search result", "tool_call_id": "call_1", "tool_name": "search"},
            {"role": "tool", "content": "Calc result", "tool_call_id": "call_2", "tool_name": "calc"},
            create_user_message("Follow up"),
        ]

        pairs = strategy._find_tool_pairs(messages)
        assert len(pairs) == 1
        assert pairs[0] == (1, 2, 3)  # Assistant message + both tool results

    def test_find_tool_pairs_incomplete_pairs(self):
        """Test finding tool pairs when some tool results are missing."""
        strategy = SmartWindowSizeContext(max_messages=10)

        messages = [
            create_user_message("Question"),
            {
                "role": "assistant",
                "content": "Tool call",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            create_user_message("Interruption"),  # No tool result
            create_assistant_message("Response"),
        ]

        pairs = strategy._find_tool_pairs(messages)
        assert len(pairs) == 0  # No complete pairs found

    def test_find_tool_pairs_no_tools(self):
        """Test finding tool pairs when no tools are present."""
        strategy = SmartWindowSizeContext(max_messages=10)

        messages = [
            create_user_message("Question"),
            create_assistant_message("Regular response"),
            create_user_message("Follow up"),
        ]

        pairs = strategy._find_tool_pairs(messages)
        assert len(pairs) == 0

    def test_get_stats(self):
        """Test SmartWindowSizeContext statistics."""
        strategy = SmartWindowSizeContext(max_messages=5, preserve_initial=2, prioritize_tools=True)
        stats = strategy.get_stats(10)

        expected = {
            "strategy": "smart_window_size",
            "max_messages": 5,
            "preserve_initial": 2,
            "prioritize_tools": True,
            "total_messages": 10,
            "messages_in_context": 5,
            "messages_dropped": 5,
        }
        assert stats == expected


class TestSmartWindowSizeContextEdgeCases:
    """Test edge cases for SmartWindowSizeContext."""

    def test_tool_pair_spans_boundary(self):
        """Test behavior when tool pair spans the preserve_initial boundary."""
        strategy = SmartWindowSizeContext(max_messages=4, preserve_initial=2, prioritize_tools=True)

        messages = [
            create_user_message("Initial 1"),
            {
                "role": "assistant",
                "content": "Tool call at boundary",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "Result", "tool_call_id": "call_123", "tool_name": "search"},
            create_user_message("After tool"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 4
        # Should include all messages since they fit exactly
        assert result == messages

    def test_multiple_tool_pairs_priority(self):
        """Test multiple tool pairs with limited space."""
        strategy = SmartWindowSizeContext(max_messages=5, preserve_initial=0, prioritize_tools=True)

        messages = [
            create_user_message("Question 1"),
            {
                "role": "assistant",
                "content": "Tool 1",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Result 1", "tool_call_id": "call_1", "tool_name": "search"},
            create_user_message("Question 2"),
            {
                "role": "assistant",
                "content": "Tool 2",
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Result 2", "tool_call_id": "call_2", "tool_name": "calc"},
            create_user_message("Final question"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 5
        # Should prioritize more recent tool pairs
        result_indices = [messages.index(msg) for msg in result]
        # Should include the second tool pair [4, 5]
        assert 4 in result_indices and 5 in result_indices

    def test_preserve_initial_with_tool_pairs(self):
        """Test preserve_initial when initial messages contain tool pairs."""
        strategy = SmartWindowSizeContext(max_messages=4, preserve_initial=3, prioritize_tools=False)

        messages = [
            create_user_message("Initial"),
            {
                "role": "assistant",
                "content": "Initial tool",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Initial result", "tool_call_id": "call_1", "tool_name": "search"},
            create_user_message("Middle"),
            create_assistant_message("Final"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 4
        # Should preserve first 3 + 1 more
        assert result[:3] == messages[:3]
        assert result[3] == messages[4]  # Final message

    def test_empty_messages_list(self):
        """Test SmartWindowSizeContext with empty messages list."""
        strategy = SmartWindowSizeContext(max_messages=5)
        result = strategy.apply_context_strategy([])
        assert result == []

    def test_single_message(self):
        """Test SmartWindowSizeContext with single message."""
        strategy = SmartWindowSizeContext(max_messages=5)
        messages = [create_user_message("Only message")]
        result = strategy.apply_context_strategy(messages)
        assert result == messages

    def test_tool_results_without_calls(self):
        """Test handling tool results without corresponding tool calls."""
        strategy = SmartWindowSizeContext(max_messages=5)

        messages = [
            create_user_message("Question"),
            {"role": "tool", "content": "Orphaned result", "tool_call_id": "call_missing", "tool_name": "search"},
            create_assistant_message("Response"),
        ]

        result = strategy.apply_context_strategy(messages)
        assert len(result) == 3
        # Should not form pairs with orphaned tool results
        assert result == messages

    def test_interleaved_conversations(self):
        """Test tool pair detection with interleaved user/assistant messages."""
        strategy = SmartWindowSizeContext(max_messages=8)

        messages = [
            create_user_message("Question 1"),
            {
                "role": "assistant",
                "content": "Tool call",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            create_user_message("Interruption"),  # Should break tool pair detection
            {"role": "tool", "content": "Result", "tool_call_id": "call_123", "tool_name": "search"},
            create_assistant_message("Response"),
        ]

        pairs = strategy._find_tool_pairs(messages)
        assert len(pairs) == 0  # No valid pairs due to interruption


class TestContextManagerWithStrategies:
    """Test ContextManager integration with different strategies."""

    def test_context_manager_with_no_strategy(self):
        """Test ContextManager with NoContextManagement strategy."""
        strategy = NoContextManagement()
        cm = ContextManager(context_management=strategy, system_prompt="Test system prompt")

        conversation_id = "test_conv"
        # Add more messages than would normally fit
        for i in range(10):
            cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        prompt = cm.get_prompt(conversation_id)
        # Should have system prompt + 10 messages
        assert len(prompt) == 11  # 1 system + 10 user messages

    def test_context_manager_with_window_strategy(self):
        """Test ContextManager with WindowSizeContext strategy."""
        strategy = WindowSizeContext(max_messages=5)
        cm = ContextManager(context_management=strategy, system_prompt="Test system prompt")

        conversation_id = "test_conv"
        # Add more messages than window size
        for i in range(10):
            cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        prompt = cm.get_prompt(conversation_id)
        # Should have system prompt + 5 messages (window size)
        assert len(prompt) == 6  # 1 system + 5 user messages
        # Should have the last 5 messages
        assert "Message 5" in prompt[1]["content"]
        assert "Message 9" in prompt[5]["content"]

    def test_context_manager_with_smart_strategy(self):
        """Test ContextManager with SmartWindowSizeContext strategy."""
        strategy = SmartWindowSizeContext(max_messages=6, preserve_initial=2, prioritize_tools=True)
        cm = ContextManager(context_management=strategy, system_prompt="Test system prompt")

        conversation_id = "test_conv"
        # Add initial messages
        cm.add_message_dict(create_user_message("Initial 1"), conversation_id)
        cm.add_message_dict(create_assistant_message("Initial response"), conversation_id)

        # Add more messages with tools
        cm.add_message_dict(create_user_message("Middle"), conversation_id)
        cm.add_message_dict(
            {
                "role": "assistant",
                "content": "Tool call",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            conversation_id,
        )
        cm.add_tool_result("search", "Search result", "call_123", conversation_id)
        cm.add_message_dict(create_user_message("After tool"), conversation_id)
        cm.add_message_dict(create_assistant_message("Final"), conversation_id)

        prompt = cm.get_prompt(conversation_id)
        # Should apply smart windowing
        assert len(prompt) == 7  # 1 system + 6 managed messages
        # Should preserve initial messages
        assert "Initial 1" in prompt[1]["content"]
        assert "Initial response" in prompt[2]["content"]

    def test_context_stats(self):
        """Test getting context statistics."""
        strategy = SmartWindowSizeContext(max_messages=5)
        cm = ContextManager(context_management=strategy)

        conversation_id = "test_conv"
        # Add messages
        for i in range(8):
            cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        stats = cm.get_context_stats(conversation_id)
        assert stats["strategy"] == "smart_window_size"
        assert stats["total_messages"] == 8
        assert stats["messages_in_context"] == 5
        assert stats["messages_dropped"] == 3

    def test_context_stats_nonexistent_conversation(self):
        """Test context statistics for non-existent conversation."""
        cm = ContextManager()
        stats = cm.get_context_stats("nonexistent")
        assert stats == {}

    def test_update_context_management(self):
        """Test updating context management strategy."""
        cm = ContextManager(system_prompt="Test system prompt")

        # Start with default (NoContextManagement)
        conversation_id = "test_conv"
        for i in range(10):
            cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        prompt_before = cm.get_prompt(conversation_id)
        assert len(prompt_before) == 11  # 1 system + 10 messages

        # Update to window strategy
        new_strategy = WindowSizeContext(max_messages=5)
        cm.update_context_management(new_strategy)

        prompt_after = cm.get_prompt(conversation_id)
        assert len(prompt_after) == 6  # 1 system + 5 messages (windowed)


class TestContextManagementPerformance:
    """Test performance aspects of context management strategies."""

    @pytest.mark.slow
    def test_large_conversation_performance(self):
        """Test performance with large conversation histories."""
        import time

        strategy = SmartWindowSizeContext(max_messages=50, preserve_initial=10, prioritize_tools=True)
        cm = ContextManager(context_management=strategy, system_prompt="Test system prompt")

        conversation_id = "large_conv"
        # Create a large conversation with mixed message types
        messages = []
        for i in range(1000):
            if i % 5 == 0:
                # Add assistant with tool calls
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Tool call {i}",
                        "tool_calls": [
                            {"id": f"call_{i}", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                        ],
                    }
                )
                messages.append(
                    {"role": "tool", "content": f"Result {i}", "tool_call_id": f"call_{i}", "tool_name": "search"}
                )
            else:
                messages.append(create_user_message(f"Message {i}"))

        # Add all messages
        start_time = time.time()
        for msg in messages:
            cm.add_message_dict(msg, conversation_id)
        add_time = time.time() - start_time

        # Get prompt (this applies context management)
        start_time = time.time()
        prompt = cm.get_prompt(conversation_id)
        prompt_time = time.time() - start_time

        # Verify results
        assert len(prompt) == 51  # 1 system + 50 managed messages
        assert add_time < 1.0  # Should be fast to add messages
        assert prompt_time < 0.5  # Context management should be fast

        # Verify context stats
        stats = cm.get_context_stats(conversation_id)
        assert stats["total_messages"] == 1200  # 1000 iterations: 800 single messages + 200 tool pairs (400 messages)
        assert stats["messages_in_context"] == 50
        assert stats["messages_dropped"] == 1150  # 1200 - 50

    @pytest.mark.slow
    def test_concurrent_conversations_performance(self):
        """Test performance with multiple concurrent conversations."""
        import time

        strategy = SmartWindowSizeContext(max_messages=20, preserve_initial=5, prioritize_tools=True)
        cm = ContextManager(context_management=strategy, system_prompt="Test system prompt")

        # Create multiple conversations
        conversation_count = 100
        messages_per_conversation = 50

        start_time = time.time()
        for conv_id in range(conversation_count):
            conversation_id = f"conv_{conv_id}"
            for msg_id in range(messages_per_conversation):
                if msg_id % 10 == 0:
                    # Add tool interaction
                    cm.add_message_dict(
                        {
                            "role": "assistant",
                            "content": f"Tool call {msg_id}",
                            "tool_calls": [
                                {
                                    "id": f"call_{conv_id}_{msg_id}",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": "{}"},
                                }
                            ],
                        },
                        conversation_id,
                    )
                    cm.add_tool_result("search", f"Result {msg_id}", f"call_{conv_id}_{msg_id}", conversation_id)
                else:
                    cm.add_message_dict(create_user_message(f"Message {msg_id}"), conversation_id)

        setup_time = time.time() - start_time

        # Get prompts for all conversations
        start_time = time.time()
        prompts = []
        for conv_id in range(conversation_count):
            conversation_id = f"conv_{conv_id}"
            prompt = cm.get_prompt(conversation_id)
            prompts.append(prompt)

        prompt_time = time.time() - start_time

        # Verify results
        assert len(prompts) == conversation_count
        assert all(len(prompt) == 21 for prompt in prompts)  # 1 system + 20 managed messages
        assert setup_time < 5.0  # Should handle 100 conversations reasonably fast
        assert prompt_time < 2.0  # Context management should be fast

        # Verify isolation
        assert len(cm.get_active_conversations()) == conversation_count

    def test_memory_usage_with_context_strategies(self):
        """Test memory usage patterns with different context strategies."""
        conversation_id = "memory_test"

        # Test with NoContextManagement (keeps all messages)
        no_context_cm = ContextManager(context_management=NoContextManagement(), system_prompt="Test system prompt")
        for i in range(1000):
            no_context_cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        # Test with WindowSizeContext (limits messages)
        window_cm = ContextManager(
            context_management=WindowSizeContext(max_messages=10), system_prompt="Test system prompt"
        )
        for i in range(1000):
            window_cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        # Test with SmartWindowSizeContext (intelligent limiting)
        smart_cm = ContextManager(
            context_management=SmartWindowSizeContext(max_messages=10), system_prompt="Test system prompt"
        )
        for i in range(1000):
            smart_cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        # All should maintain the same number of stored messages (raw history)
        assert len(no_context_cm._conversations[conversation_id]) == 1000
        assert len(window_cm._conversations[conversation_id]) == 1000
        assert len(smart_cm._conversations[conversation_id]) == 1000

        # But prompt generation should respect limits
        no_context_prompt = no_context_cm.get_prompt(conversation_id)
        window_prompt = window_cm.get_prompt(conversation_id)
        smart_prompt = smart_cm.get_prompt(conversation_id)

        assert len(no_context_prompt) == 1001  # 1 system + 1000 messages
        assert len(window_prompt) == 11  # 1 system + 10 messages
        assert len(smart_prompt) == 11  # 1 system + 10 messages

    def test_context_strategy_switching_performance(self):
        """Test performance when switching between context strategies."""
        import time

        cm = ContextManager(system_prompt="Test system prompt")
        conversation_id = "switch_test"

        # Add many messages
        for i in range(500):
            cm.add_message_dict(create_user_message(f"Message {i}"), conversation_id)

        # Test switching strategies
        strategies = [
            NoContextManagement(),
            WindowSizeContext(max_messages=20),
            SmartWindowSizeContext(max_messages=20, preserve_initial=5, prioritize_tools=True),
        ]

        switch_times = []
        for strategy in strategies:
            start_time = time.time()
            cm.update_context_management(strategy)
            cm.get_prompt(conversation_id)
            switch_time = time.time() - start_time
            switch_times.append(switch_time)

        # All switches should be fast
        assert all(switch_time < 0.1 for switch_time in switch_times)

        # Final prompt should use the last strategy (SmartWindowSizeContext)
        final_prompt = cm.get_prompt(conversation_id)
        assert len(final_prompt) == 21  # 1 system + 20 messages


class TestContextManagementConcurrency:
    """Test concurrent access to context management."""

    @pytest.mark.asyncio
    async def test_concurrent_message_addition(self):
        """Test adding messages concurrently to different conversations."""
        import asyncio

        cm = ContextManager(
            context_management=SmartWindowSizeContext(max_messages=10), system_prompt="Test system prompt"
        )

        async def add_messages_to_conversation(conv_id, message_count):
            for i in range(message_count):
                cm.add_message_dict(create_user_message(f"Conv {conv_id} Message {i}"), f"conv_{conv_id}")
                if i % 5 == 0:
                    await asyncio.sleep(0.001)  # Small delay to allow interleaving

        # Run multiple conversations concurrently
        tasks = []
        for conv_id in range(10):
            task = add_messages_to_conversation(conv_id, 50)
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Verify all conversations were created correctly
        assert len(cm.get_active_conversations()) == 10

        # Verify each conversation has the correct number of messages
        for conv_id in range(10):
            conversation_id = f"conv_{conv_id}"
            history = cm.get_conversation_history(conversation_id)
            assert len(history) == 50

            # Verify prompt generation works correctly
            prompt = cm.get_prompt(conversation_id)
            assert len(prompt) == 11  # 1 system + 10 managed messages

    @pytest.mark.asyncio
    async def test_concurrent_prompt_generation(self):
        """Test generating prompts concurrently for different conversations."""
        import asyncio

        cm = ContextManager(
            context_management=SmartWindowSizeContext(max_messages=20), system_prompt="Test system prompt"
        )

        # Setup multiple conversations
        for conv_id in range(20):
            conversation_id = f"conv_{conv_id}"
            for msg_id in range(100):
                if msg_id % 10 == 0:
                    cm.add_message_dict(
                        {
                            "role": "assistant",
                            "content": f"Tool call {msg_id}",
                            "tool_calls": [
                                {
                                    "id": f"call_{conv_id}_{msg_id}",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": "{}"},
                                }
                            ],
                        },
                        conversation_id,
                    )
                    cm.add_tool_result("search", f"Result {msg_id}", f"call_{conv_id}_{msg_id}", conversation_id)
                else:
                    cm.add_message_dict(create_user_message(f"Message {msg_id}"), conversation_id)

        async def get_prompt_for_conversation(conv_id):
            conversation_id = f"conv_{conv_id}"
            prompt = cm.get_prompt(conversation_id)
            return len(prompt), conversation_id

        # Get prompts concurrently
        tasks = []
        for conv_id in range(20):
            task = get_prompt_for_conversation(conv_id)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all prompts were generated correctly
        assert len(results) == 20
        assert all(prompt_len == 21 for prompt_len, _ in results)  # 1 system + 20 managed messages

    def test_thread_safety_simulation(self):
        """Test thread safety by simulating concurrent operations."""
        import threading
        import time

        cm = ContextManager(
            context_management=SmartWindowSizeContext(max_messages=15), system_prompt="Test system prompt"
        )
        results = []
        errors = []

        def worker(worker_id):
            try:
                conversation_id = f"thread_{worker_id}"

                # Add messages
                for i in range(50):
                    cm.add_message_dict(create_user_message(f"Worker {worker_id} Message {i}"), conversation_id)
                    if i % 20 == 0:
                        time.sleep(0.001)  # Small delay

                # Get prompts
                for _ in range(10):
                    prompt = cm.get_prompt(conversation_id)
                    results.append((worker_id, len(prompt)))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify results
        assert len(results) == 50  # 5 workers × 10 prompts each
        assert all(prompt_len == 16 for _, prompt_len in results)  # 1 system + 15 managed messages

        # Verify conversations were created
        assert len(cm.get_active_conversations()) == 5
