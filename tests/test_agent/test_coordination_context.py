"""Tests for CoordinationContextManager class."""

import pytest
from unittest.mock import Mock

from spade.message import Message
from spade_llm.agent.coordinator_agent import CoordinationContextManager
from spade_llm.context._types import create_user_message


class TestCoordinationContextManagerInitialization:
    """Test CoordinationContextManager initialization."""

    def test_initialization_with_coordination_session(
        self, coordination_session_id, subagent_ids
    ):
        """Test initialization with coordination parameters."""
        ccm = CoordinationContextManager(
            coordination_session=coordination_session_id,
            subagent_ids=set(subagent_ids)
        )

        assert ccm.coordination_session == coordination_session_id
        assert ccm.subagent_ids == set(subagent_ids)
        # Check parent class attributes
        assert ccm._conversations == {}
        assert ccm._active_conversations == set()
        assert ccm._current_conversation_id is None

    def test_initialization_with_empty_subagent_list(self, coordination_session_id):
        """Test initialization with empty subagent set."""
        ccm = CoordinationContextManager(
            coordination_session=coordination_session_id,
            subagent_ids=set()
        )

        assert ccm.subagent_ids == set()
        assert ccm.coordination_session == coordination_session_id

    def test_initialization_with_system_prompt(
        self, coordination_session_id, subagent_ids
    ):
        """Test system prompt propagation to parent."""
        system_prompt = "Test coordinator system prompt"
        ccm = CoordinationContextManager(
            coordination_session=coordination_session_id,
            subagent_ids=set(subagent_ids),
            system_prompt=system_prompt
        )

        assert ccm._system_prompt == system_prompt

    def test_initialization_with_context_management_strategy(
        self, coordination_session_id, subagent_ids
    ):
        """Test context management strategy integration."""
        from spade_llm.context.management import WindowSizeContext

        strategy = WindowSizeContext(max_messages=10)
        ccm = CoordinationContextManager(
            coordination_session=coordination_session_id,
            subagent_ids=set(subagent_ids),
            context_management=strategy
        )

        assert ccm.context_management == strategy

    def test_sanitize_jid_for_name_assignment(
        self, coordination_session_id, subagent_ids
    ):
        """Test that sanitization function is available."""
        ccm = CoordinationContextManager(
            coordination_session=coordination_session_id,
            subagent_ids=set(subagent_ids)
        )

        assert hasattr(ccm, '_sanitize_jid_for_name')
        assert callable(ccm._sanitize_jid_for_name)


class TestConversationIDOverrideLogic:
    """Test conversation ID override logic for coordination."""

    def test_conversation_id_override_subagent_sender(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test coordination_session used when sender is subagent."""
        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = "some_thread"

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        assert conv_id == coordination_session_id

    def test_conversation_id_override_subagent_receiver(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test coordination_session used when receiver is subagent."""
        msg = Mock(spec=Message)
        msg.sender = "coordinator@localhost"
        msg.to = "subagent2@localhost"
        msg.thread = "some_thread"

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        assert conv_id == coordination_session_id

    def test_conversation_id_override_matching_thread(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test coordination_session used when thread matches."""
        msg = Mock(spec=Message)
        msg.sender = "other@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = coordination_session_id

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        assert conv_id == coordination_session_id

    def test_conversation_id_external_message(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test external messages use standard logic."""
        msg = Mock(spec=Message)
        msg.sender = "external_user@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = "external_thread_789"

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        assert conv_id == "external_thread_789"
        assert conv_id != coordination_session_id

    def test_conversation_id_external_with_thread(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test external messages with custom thread."""
        msg = Mock(spec=Message)
        msg.sender = "user@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = "custom_thread"

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        assert conv_id == "custom_thread"

    def test_conversation_id_external_without_thread(
        self, coordination_context_manager
    ):
        """Test external messages without thread."""
        msg = Mock(spec=Message)
        msg.sender = "user@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = None

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        assert conv_id.startswith("user_coordinator_")

    def test_conversation_id_mixed_scenarios(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test multiple messages with different routing."""
        # Message 1: Subagent to coordinator
        msg1 = Mock(spec=Message)
        msg1.sender = "subagent1@localhost"
        msg1.to = "coordinator@localhost"
        msg1.thread = "thread1"

        conv_id1 = coordination_context_manager._get_coordination_conversation_id(msg1)
        assert conv_id1 == coordination_session_id

        # Message 2: Coordinator to subagent
        msg2 = Mock(spec=Message)
        msg2.sender = "coordinator@localhost"
        msg2.to = "subagent2@localhost"
        msg2.thread = "thread2"

        conv_id2 = coordination_context_manager._get_coordination_conversation_id(msg2)
        assert conv_id2 == coordination_session_id

        # Message 3: External to coordinator
        msg3 = Mock(spec=Message)
        msg3.sender = "user@localhost"
        msg3.to = "coordinator@localhost"
        msg3.thread = "external_thread"

        conv_id3 = coordination_context_manager._get_coordination_conversation_id(msg3)
        assert conv_id3 == "external_thread"

    def test_conversation_id_with_partial_jid_match(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that exact JID matching is required (not substring)."""
        msg = Mock(spec=Message)
        msg.sender = "subagent123@localhost"  # Not in subagent_ids
        msg.to = "coordinator@localhost"
        msg.thread = "some_thread"

        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)

        # Should NOT be treated as subagent message
        assert conv_id == "some_thread"
        assert conv_id != coordination_session_id


class TestMessageAdditionAndContextUpdates:
    """Test message addition and context update functionality."""

    def test_add_message_uses_coordination_logic(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that add_message uses override logic."""
        msg = Mock(spec=Message)
        msg.body = "Test message from subagent"
        msg.sender = "subagent1@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = "any_thread"

        # Add message without explicit conversation_id
        coordination_context_manager.add_message(msg)

        # Should be added to coordination_session
        assert coordination_session_id in coordination_context_manager._conversations
        assert len(coordination_context_manager._conversations[coordination_session_id]) == 1

    def test_add_message_explicit_conversation_id(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that explicit conversation_id parameter is respected."""
        msg = Mock(spec=Message)
        msg.body = "Test message"
        msg.sender = "subagent1@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = coordination_session_id

        custom_conv_id = "custom_conversation"
        coordination_context_manager.add_message(msg, custom_conv_id)

        # Should be added to custom_id, not coordination_session
        assert custom_conv_id in coordination_context_manager._conversations
        assert len(coordination_context_manager._conversations[custom_conv_id]) == 1

    def test_add_coordination_message_dict(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test coordination-style commands via the current message-dict API."""
        target_agent = "subagent1@localhost"
        command = "do task"
        message = create_user_message(
            f"[TO: {target_agent}] {command}",
            name=coordination_context_manager._sanitize_jid_for_name(target_agent),
        )

        coordination_context_manager.add_message_dict(message, coordination_session_id)

        # Check message added to coordination_session
        assert coordination_session_id in coordination_context_manager._conversations
        messages = coordination_context_manager._conversations[coordination_session_id]
        assert len(messages) == 1

        # Check message format
        msg = messages[0]
        assert msg["role"] == "user"
        assert f"[TO: {target_agent}]" in msg["content"]
        assert command in msg["content"]
        assert msg["name"] == "subagent1@localhost"

    def test_add_message_dict_custom_conversation(
        self, coordination_context_manager
    ):
        """Test override conversation_id when adding a message dict."""
        custom_conv_id = "custom_coordination"
        message = create_user_message(
            "test command",
            name=coordination_context_manager._sanitize_jid_for_name(
                "subagent1@localhost"
            ),
        )

        coordination_context_manager.add_message_dict(message, custom_conv_id)

        # Should be added to specified conversation
        assert custom_conv_id in coordination_context_manager._conversations
        assert len(coordination_context_manager._conversations[custom_conv_id]) == 1
        assert coordination_context_manager._conversations[custom_conv_id][0] == message

    def test_multiple_subagent_messages_same_conversation(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that all subagent messages share context."""
        # Add messages from different subagents
        for i, agent_id in enumerate(["subagent1@localhost", "subagent2@localhost", "subagent3@localhost"]):
            msg = Mock(spec=Message)
            msg.body = f"Message {i} from {agent_id}"
            msg.sender = agent_id
            msg.to = "coordinator@localhost"
            msg.thread = f"thread_{i}"

            coordination_context_manager.add_message(msg)

        # All should be in same coordination_session conversation
        messages = coordination_context_manager._conversations[coordination_session_id]
        assert len(messages) == 3

    def test_message_metadata_preservation(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that SPADE message metadata is preserved."""
        msg = Mock(spec=Message)
        msg.body = "Test message"
        msg.sender = "subagent1@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = coordination_session_id

        coordination_context_manager.add_message(msg)

        stored_msg = coordination_context_manager._conversations[coordination_session_id][0]
        assert "sender" in stored_msg
        assert "receiver" in stored_msg
        assert "thread" in stored_msg
        assert stored_msg["sender"] == str(msg.sender)
        assert stored_msg["receiver"] == str(msg.to)
        assert stored_msg["thread"] == msg.thread


class TestContextIsolationAndVisibility:
    """Test context isolation and visibility patterns."""

    def test_coordinator_sees_all_subagent_messages(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that coordinator has full context visibility."""
        # Simulate full workflow
        messages_to_add = [
            # User to coordinator
            {"body": "Please coordinate task", "sender": "user@localhost", "to": "coordinator@localhost", "thread": "ext"},
            # Coordinator to subagent1
            {"body": "Do step 1", "sender": "coordinator@localhost", "to": "subagent1@localhost", "thread": coordination_session_id},
            # Subagent1 to coordinator
            {"body": "Step 1 done", "sender": "subagent1@localhost", "to": "coordinator@localhost", "thread": coordination_session_id},
            # Coordinator to subagent2
            {"body": "Do step 2", "sender": "coordinator@localhost", "to": "subagent2@localhost", "thread": coordination_session_id},
            # Subagent2 to coordinator
            {"body": "Step 2 done", "sender": "subagent2@localhost", "to": "coordinator@localhost", "thread": coordination_session_id},
        ]

        for msg_data in messages_to_add:
            msg = Mock(spec=Message)
            msg.body = msg_data["body"]
            msg.sender = msg_data["sender"]
            msg.to = msg_data["to"]
            msg.thread = msg_data["thread"]
            coordination_context_manager.add_message(msg)

        # Check coordination_session has subagent messages
        coord_messages = coordination_context_manager._conversations[coordination_session_id]
        # Should have 4 messages (excluding external user message)
        assert len(coord_messages) == 4

        # Verify prompt includes all coordination messages
        prompt = coordination_context_manager.get_prompt(coordination_session_id)
        # Filter out system messages
        user_and_assistant_msgs = [m for m in prompt if m["role"] in ["user", "assistant"]]
        assert len(user_and_assistant_msgs) == 4

    def test_external_conversations_isolated(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test that non-coordination conversations remain separate."""
        # Add coordination messages
        msg1 = Mock(spec=Message)
        msg1.body = "Coordination message"
        msg1.sender = "subagent1@localhost"
        msg1.to = "coordinator@localhost"
        msg1.thread = coordination_session_id
        coordination_context_manager.add_message(msg1)

        # Add external messages
        msg2 = Mock(spec=Message)
        msg2.body = "External message"
        msg2.sender = "external@localhost"
        msg2.to = "coordinator@localhost"
        msg2.thread = "external_thread"
        coordination_context_manager.add_message(msg2)

        # Should have two separate conversations
        assert len(coordination_context_manager._conversations) == 2
        assert coordination_session_id in coordination_context_manager._conversations
        assert "external_thread" in coordination_context_manager._conversations

        # No cross-contamination
        assert len(coordination_context_manager._conversations[coordination_session_id]) == 1
        assert len(coordination_context_manager._conversations["external_thread"]) == 1

    def test_subagent_context_isolation_simulation(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test simulated subagent's isolated view."""
        from spade_llm.context import ContextManager

        # Add messages to coordinator's context
        for msg_data in [
            {"body": "Command for agent1", "sender": "coordinator@localhost", "to": "subagent1@localhost"},
            {"body": "Response from agent1", "sender": "subagent1@localhost", "to": "coordinator@localhost"},
            {"body": "Command for agent2", "sender": "coordinator@localhost", "to": "subagent2@localhost"},
        ]:
            msg = Mock(spec=Message)
            msg.body = msg_data["body"]
            msg.sender = msg_data["sender"]
            msg.to = msg_data["to"]
            msg.thread = coordination_session_id
            coordination_context_manager.add_message(msg)

        # Simulate subagent1's isolated context (would only see its own messages)
        subagent_context = ContextManager()
        msg_for_subagent = Mock(spec=Message)
        msg_for_subagent.body = "Command for agent1"
        msg_for_subagent.sender = "coordinator@localhost"
        msg_for_subagent.to = "subagent1@localhost"
        msg_for_subagent.thread = coordination_session_id
        subagent_context.add_message(msg_for_subagent, coordination_session_id)

        # Coordinator sees 3 messages
        coordinator_messages = coordination_context_manager._conversations[coordination_session_id]
        assert len(coordinator_messages) == 3

        # Subagent sees only 1 message
        subagent_messages = subagent_context._conversations[coordination_session_id]
        assert len(subagent_messages) == 1

    def test_concurrent_coordination_sessions(
        self, coordination_session_id, subagent_ids
    ):
        """Test that multiple coordination sessions don't interfere."""
        # Create two separate coordination context managers
        ccm1 = CoordinationContextManager(
            coordination_session="session_1",
            subagent_ids=set(subagent_ids)
        )
        ccm2 = CoordinationContextManager(
            coordination_session="session_2",
            subagent_ids=set(subagent_ids)
        )

        # Add messages to session 1
        msg1 = Mock(spec=Message)
        msg1.body = "Session 1 message"
        msg1.sender = "subagent1@localhost"
        msg1.to = "coordinator@localhost"
        msg1.thread = "session_1"
        ccm1.add_message(msg1)

        # Add messages to session 2
        msg2 = Mock(spec=Message)
        msg2.body = "Session 2 message"
        msg2.sender = "subagent1@localhost"
        msg2.to = "coordinator@localhost"
        msg2.thread = "session_2"
        ccm2.add_message(msg2)

        # Each maintains separate contexts
        assert len(ccm1._conversations["session_1"]) == 1
        assert len(ccm2._conversations["session_2"]) == 1
        assert "session_2" not in ccm1._conversations
        assert "session_1" not in ccm2._conversations


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_message_without_thread_attribute(
        self, coordination_context_manager, coordination_session_id
    ):
        """Test handling of messages without thread field."""
        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = None

        # Should use coordination_session because sender is subagent
        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)
        assert conv_id == coordination_session_id

    def test_message_with_hasattr_to_check(
        self, coordination_context_manager
    ):
        """Test message 'to' attribute check with hasattr."""
        msg = Mock(spec=Message)
        msg.sender = "external@localhost"
        # Deliberately don't set 'to' attribute
        if hasattr(msg, 'to'):
            delattr(msg, 'to')
        msg.thread = "test_thread"

        # Should handle gracefully
        conv_id = coordination_context_manager._get_coordination_conversation_id(msg)
        assert conv_id == "test_thread"

    def test_add_message_dict_with_special_characters(
        self, coordination_context_manager
    ):
        """Test message dict content and sanitized names with special characters."""
        agent_id = "agent.with_special-chars@domain.com"
        command = "Do task with 'quotes' and \"double quotes\""
        message = create_user_message(
            command,
            name=coordination_context_manager._sanitize_jid_for_name(agent_id),
        )

        coordination_context_manager.add_message_dict(
            message, coordination_context_manager.coordination_session
        )

        messages = list(coordination_context_manager._conversations.values())[0]
        assert len(messages) == 1
        assert command in messages[0]["content"]
        assert messages[0]["name"] == agent_id
