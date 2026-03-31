"""Tests for LLMBehaviour class."""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from spade_llm.behaviour import LLMBehaviour
from spade_llm.behaviour.llm_behaviour import ConversationState
from spade_llm.context import ContextManager
from spade_llm.routing import RoutingResponse
from tests.conftest import MockLLMProvider


class TestLLMBehaviourInitialization:
    """Test LLMBehaviour initialization."""

    def test_init_minimal(self, mock_llm_provider):
        """Test initialization with minimal parameters."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)

        assert behaviour.provider == mock_llm_provider
        assert isinstance(behaviour.context, ContextManager)
        assert behaviour.reply_to is None
        assert behaviour.routing_function is None
        assert behaviour.termination_markers == ["<TASK_COMPLETE>", "<END>", "<DONE>"]
        assert behaviour.max_interactions_per_conversation is None
        assert behaviour.on_conversation_end is None
        assert behaviour.tools == []
        assert isinstance(behaviour._active_conversations, dict)
        assert isinstance(behaviour._processed_messages, set)

    def test_init_full_parameters(self, mock_llm_provider, context_manager, mock_simple_tool):
        """Test initialization with all parameters."""

        def routing_func(msg, response, context):
            return "target@localhost"

        def end_callback(conv_id, reason):
            pass

        behaviour = LLMBehaviour(
            llm_provider=mock_llm_provider,
            reply_to="reply@localhost",
            routing_function=routing_func,
            context_manager=context_manager,
            termination_markers=["<STOP>", "<FINISHED>"],
            max_interactions_per_conversation=5,
            on_conversation_end=end_callback,
            tools=[mock_simple_tool],
        )

        assert behaviour.provider == mock_llm_provider
        assert behaviour.context == context_manager
        assert behaviour.reply_to == "reply@localhost"
        assert behaviour.routing_function == routing_func
        assert behaviour.termination_markers == ["<STOP>", "<FINISHED>"]
        assert behaviour.max_interactions_per_conversation == 5
        assert behaviour.on_conversation_end == end_callback
        assert len(behaviour.tools) == 1
        assert behaviour.tools[0] == mock_simple_tool


class TestLLMBehaviourMessageProcessing:
    """Test message processing functionality."""

    @pytest.mark.asyncio
    async def test_run_no_message(self, mock_llm_provider):
        """Test run method when no message is received."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=None)

        # Should return without error when no message
        await behaviour.run()

        behaviour.receive.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_run_duplicate_message(self, mock_llm_provider, mock_message):
        """Test run method skips duplicate messages."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        # Process message once
        await behaviour.run()

        # Process same message again
        await behaviour.run()

        # Should have been processed only once
        assert len(behaviour._processed_messages) == 1
        assert mock_message.id in behaviour._processed_messages

    @pytest.mark.asyncio
    async def test_run_basic_message_processing(self, mock_llm_provider, mock_message):
        """Test basic message processing flow."""
        mock_llm_provider.responses = ["Hello! How can I help you?"]

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have processed the message
        assert mock_message.id in behaviour._processed_messages

        # Should have created conversation state
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        assert conv_id in behaviour._active_conversations

        # Should have sent response
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert sent_message.body == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_run_with_reply_to(self, mock_llm_provider, mock_message):
        """Test message processing with reply_to specified."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider, reply_to="custom@localhost")
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send to reply_to address
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert sent_message.to == "custom@localhost"

    @pytest.mark.asyncio
    async def test_run_with_routing_function(self, mock_llm_provider, mock_message):
        """Test message processing with routing function."""

        def routing_func(original_msg, response, context):
            return "routed@localhost"

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider, routing_function=routing_func)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send to routed address
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert sent_message.to == "routed@localhost"

    @pytest.mark.asyncio
    async def test_run_with_routing_response_object(self, mock_llm_provider, mock_message):
        """Test message processing with RoutingResponse object."""

        def routing_func(original_msg, response, context):
            return RoutingResponse(
                recipients=["recipient1@localhost", "recipient2@localhost"],
                transform=lambda text: f"Transformed: {text}",
                metadata={"key": "value"},
            )

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider, routing_function=routing_func)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send to both recipients
        assert behaviour.send.call_count == 2

        # Check first message
        first_call = behaviour.send.call_args_list[0][0][0]
        assert first_call.to == "recipient1@localhost"
        assert first_call.body.startswith("Transformed:")

        # Check second message
        second_call = behaviour.send.call_args_list[1][0][0]
        assert second_call.to == "recipient2@localhost"
        assert second_call.body.startswith("Transformed:")


class TestLLMBehaviourConversationManagement:
    """Test conversation management functionality."""

    @pytest.mark.asyncio
    async def test_conversation_state_creation(self, mock_llm_provider, mock_message):
        """Test that conversation state is created properly."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        conversation = behaviour._active_conversations[conv_id]

        assert conversation["state"] == ConversationState.ACTIVE
        assert conversation["interaction_count"] == 1
        assert "start_time" in conversation
        assert "last_activity" in conversation

    @pytest.mark.asyncio
    async def test_termination_markers(self, mock_llm_provider, mock_message):
        """Test conversation termination based on markers."""
        mock_llm_provider.responses = ["Task completed <DONE>"]

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider, termination_markers=["<DONE>"])
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have terminated conversation
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        conversation = behaviour._active_conversations[conv_id]
        assert conversation["state"] == ConversationState.COMPLETED

    @pytest.mark.asyncio
    async def test_completed_conversation_not_processed(self, mock_llm_provider, mock_message):
        """Test that completed conversations are not processed further."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        # Manually set conversation as completed
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        behaviour._active_conversations[conv_id] = {
            "state": ConversationState.COMPLETED,
            "interaction_count": 1,
            "start_time": time.time(),
            "last_activity": time.time(),
        }

        await behaviour.run()

        # Should not have sent any message
        behaviour.send.assert_not_called()

    def test_reset_conversation(self, mock_llm_provider):
        """Test resetting a conversation."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)

        # Create a conversation
        conv_id = "test_conversation"
        behaviour._active_conversations[conv_id] = {
            "state": ConversationState.COMPLETED,
            "interaction_count": 5,
            "start_time": time.time() - 100,
            "last_activity": time.time() - 50,
        }

        # Reset it
        result = behaviour.reset_conversation(conv_id)

        assert result is True
        conversation = behaviour._active_conversations[conv_id]
        assert conversation["state"] == ConversationState.ACTIVE
        assert conversation["interaction_count"] == 0

    def test_reset_nonexistent_conversation(self, mock_llm_provider):
        """Test resetting a non-existent conversation."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)

        result = behaviour.reset_conversation("nonexistent")

        assert result is False

    def test_get_conversation_state(self, mock_llm_provider):
        """Test getting conversation state."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)

        # Create a conversation
        conv_id = "test_conversation"
        expected_state = {
            "state": ConversationState.ACTIVE,
            "interaction_count": 3,
            "start_time": time.time(),
            "last_activity": time.time(),
        }
        behaviour._active_conversations[conv_id] = expected_state

        result = behaviour.get_conversation_state(conv_id)

        assert result == expected_state

    def test_get_nonexistent_conversation_state(self, mock_llm_provider):
        """Test getting state for non-existent conversation."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)

        result = behaviour.get_conversation_state("nonexistent")

        assert result is None


class TestLLMBehaviourToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_tool_calling_flow(self, mock_simple_tool, mock_message):
        """Test complete tool calling flow."""
        # Create provider that returns tool calls first, then final response
        provider = MockLLMProvider(
            responses=["Final response after tool use"],
            tool_calls=[[{"id": "call_123", "name": "simple_tool", "arguments": {"text": "test input"}}]],
        )

        behaviour = LLMBehaviour(llm_provider=provider, tools=[mock_simple_tool])
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have processed tool call and sent final response
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert sent_message.body == "Final response after tool use"

        # Should have called LLM twice (once for tool call, once for final response)
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_not_found(self, mock_message):
        """Test handling when requested tool is not found."""
        provider = MockLLMProvider(
            responses=["Final response after error"],
            tool_calls=[[{"id": "call_123", "name": "nonexistent_tool", "arguments": {"param": "value"}}]],
        )

        behaviour = LLMBehaviour(
            llm_provider=provider,
            tools=[],  # No tools available
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should still send response despite tool error
        behaviour.send.assert_called_once()

        # Should have added error to context
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        history = behaviour.context.get_conversation_history(conv_id)

        # Should have user message, assistant tool call, tool error, and final response
        assert len(history) >= 3

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, mock_error_tool, mock_message):
        """Test handling tool execution errors."""
        provider = MockLLMProvider(
            responses=["Final response after tool error"],
            tool_calls=[[{"id": "call_123", "name": "error_tool", "arguments": {}}]],
        )

        behaviour = LLMBehaviour(llm_provider=provider, tools=[mock_error_tool])
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should still complete and send response
        behaviour.send.assert_called_once()

        # Tool error should be recorded in context
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        history = behaviour.context.get_conversation_history(conv_id)

        # Find tool result message
        tool_messages = [msg for msg in history if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "error" in tool_messages[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, mock_simple_tool, mock_async_tool, mock_message):
        """Test handling multiple tool calls in sequence."""
        provider = MockLLMProvider(
            responses=["Final response with both tools"],
            tool_calls=[
                [
                    {"id": "call_1", "name": "simple_tool", "arguments": {"text": "first"}},
                    {"id": "call_2", "name": "async_tool", "arguments": {"number": 10}},
                ]
            ],
        )

        behaviour = LLMBehaviour(llm_provider=provider, tools=[mock_simple_tool, mock_async_tool])
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should complete successfully
        behaviour.send.assert_called_once()

        # Both tools should have been executed
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        history = behaviour.context.get_conversation_history(conv_id)

        tool_messages = [msg for msg in history if msg.get("role") == "tool"]
        assert len(tool_messages) == 2

    @pytest.mark.asyncio
    async def test_max_tool_iterations(self, mock_simple_tool, mock_message):
        """Test max tool iterations limit."""
        # Create provider that always returns tool calls
        provider = MockLLMProvider(
            responses=["Forced final response"],
            tool_calls=[
                [{"id": f"call_{i}", "name": "simple_tool", "arguments": {"text": f"iteration_{i}"}}] for i in range(25)
            ],  # More than max iterations
        )

        behaviour = LLMBehaviour(llm_provider=provider, tools=[mock_simple_tool])
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should eventually complete with forced response
        behaviour.send.assert_called_once()

        # Should have stopped at max iterations (20)
        assert provider.call_count <= 21  # 20 tool iterations + 1 final

    def test_register_tool(self, mock_llm_provider, mock_simple_tool):
        """Test registering a tool with the behaviour."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)

        assert len(behaviour.tools) == 0

        behaviour.register_tool(mock_simple_tool)

        assert len(behaviour.tools) == 1
        assert behaviour.tools[0] == mock_simple_tool

    def test_get_tools(self, mock_llm_provider, mock_simple_tool, mock_async_tool):
        """Test getting tools from the behaviour."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider, tools=[mock_simple_tool, mock_async_tool])

        tools = behaviour.get_tools()

        assert len(tools) == 2
        assert mock_simple_tool in tools
        assert mock_async_tool in tools


class TestLLMBehaviourErrorHandling:
    """Test error handling in LLMBehaviour."""

    @pytest.mark.asyncio
    async def test_llm_provider_error(self, mock_llm_provider_error, mock_message):
        """Test handling LLM provider errors."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider_error)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send error message
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "Error processing your message" in sent_message.body

        # Conversation should be marked as error
        conv_id = mock_message.thread or f"{mock_message.sender}_{mock_message.to}"
        conversation = behaviour._active_conversations[conv_id]
        assert conversation["state"] == ConversationState.ERROR

    @pytest.mark.asyncio
    async def test_send_error_handling(self, mock_llm_provider, mock_message):
        """Test handling errors when sending messages."""
        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock(side_effect=Exception("Send error"))

        # Should not raise exception, just log error
        await behaviour.run()

        # Send should have been attempted
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_llm_response_handling(self, mock_message):
        """Test handling empty LLM response."""
        provider = MockLLMProvider(responses=[""])

        behaviour = LLMBehaviour(llm_provider=provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send fallback message (empty string is falsy, so fallback is used)
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "I'm sorry, I couldn't complete this request" in sent_message.body

    @pytest.mark.asyncio
    async def test_none_llm_response_handling(self, mock_message):
        """Test handling None LLM response."""
        provider = MockLLMProvider(responses=[None])

        behaviour = LLMBehaviour(llm_provider=provider)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send fallback message
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "I'm sorry, I couldn't complete this request" in sent_message.body


class TestLLMBehaviourEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_message_without_thread(self, mock_llm_provider):
        """Test handling message without thread ID."""
        # Create message without thread
        msg = Mock()
        msg.body = "Test message"
        msg.sender = "sender@localhost"
        msg.to = "receiver@localhost"
        msg.thread = None  # No thread
        msg.id = "msg_123"
        msg.metadata = {}

        reply = Mock()
        reply.to = msg.sender
        reply.body = ""
        reply.thread = None
        reply.set_metadata = Mock()
        msg.make_reply = Mock(return_value=reply)

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should create conversation ID from sender_receiver_uuid format
        assert any(key.startswith("sender_receiver_") for key in behaviour._active_conversations)

    @pytest.mark.asyncio
    async def test_message_with_empty_body(self, mock_llm_provider):
        """Test handling message with empty body."""
        msg = Mock()
        msg.body = ""  # Empty body
        msg.sender = "sender@localhost"
        msg.to = "receiver@localhost"
        msg.thread = "test_thread"
        msg.id = "msg_123"
        msg.metadata = {}

        reply = Mock()
        reply.to = msg.sender
        reply.body = ""
        reply.thread = msg.thread
        reply.set_metadata = Mock()
        msg.make_reply = Mock(return_value=reply)

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should still process successfully
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_rapid_message_processing(self, mock_llm_provider):
        """Test processing multiple messages rapidly."""
        messages = []
        for i in range(5):
            msg = Mock()
            msg.body = f"Message {i}"
            msg.sender = "sender@localhost"
            msg.to = "receiver@localhost"
            msg.thread = "rapid_test"
            msg.id = f"msg_{i}"
            msg.metadata = {}

            reply = Mock()
            reply.to = msg.sender
            reply.body = ""
            reply.thread = msg.thread
            reply.set_metadata = Mock()
            msg.make_reply = Mock(return_value=reply)

            messages.append(msg)

        behaviour = LLMBehaviour(llm_provider=mock_llm_provider)
        behaviour.send = AsyncMock()

        # Process all messages
        for msg in messages:
            behaviour.receive = AsyncMock(return_value=msg)
            await behaviour.run()

        # All messages should have been processed
        assert len(behaviour._processed_messages) == 5
        assert behaviour.send.call_count == 5

        # Conversation should have correct interaction count
        conversation = behaviour._active_conversations["rapid_test"]
        assert conversation["interaction_count"] == 5
