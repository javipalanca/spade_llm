"""Tests for CoordinatorAgent class."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from spade.message import Message

from spade_llm.agent.coordinator_agent import CoordinationContextManager, CoordinatorAgent


class TestCoordinatorAgentInitialization:
    """Test CoordinatorAgent initialization."""

    def test_initialization_minimal(self, mock_llm_provider, subagent_ids):
        """Test minimal valid initialization."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        assert agent.jid == "coordinator@localhost"
        assert agent.password == "password"
        assert agent.provider == mock_llm_provider
        assert agent.subagent_ids == set(subagent_ids)
        assert agent.coordination_session.startswith("coordination_")
        assert agent.routing_function is not None
        assert isinstance(agent.context, CoordinationContextManager)
        assert len(agent.agent_status) == 3
        assert all(status == "idle" for status in agent.agent_status.values())
        assert agent._original_requester is None
        assert agent.termination_markers == ["<TASK_COMPLETE>", "<END>", "<DONE>"]

    def test_initialization_with_custom_session(self, mock_llm_provider, subagent_ids):
        """Test custom coordination_session parameter."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session="custom_session",
        )

        assert agent.coordination_session == "custom_session"
        assert agent.context.coordination_session == "custom_session"

    def test_initialization_with_custom_routing_function(self, mock_llm_provider, subagent_ids):
        """Test custom routing function override."""

        def custom_routing(msg, response, context):
            return "custom_target@localhost"

        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            routing_function=custom_routing,
        )

        assert agent.routing_function == custom_routing

    def test_initialization_with_custom_system_prompt(self, mock_llm_provider, subagent_ids):
        """Test custom system prompt override."""
        custom_prompt = "Custom coordinator prompt"
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            system_prompt=custom_prompt,
        )

        assert agent.context._system_prompt == custom_prompt

    def test_initialization_empty_subagent_ids_raises_error(self, mock_llm_provider):
        """Test validation of required subagent_ids."""
        with pytest.raises(ValueError, match="subagent_ids cannot be empty"):
            CoordinatorAgent(
                jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=[]
            )

    def test_context_override_injection(self, mock_llm_provider, subagent_ids):
        """Test CoordinationContextManager injection."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session="test_session",
        )

        assert isinstance(agent.context, CoordinationContextManager)
        assert agent.context.coordination_session == "test_session"
        assert agent.context.subagent_ids == set(subagent_ids)

    def test_termination_markers_default(self, mock_llm_provider, subagent_ids):
        """Test default termination markers."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        assert agent.termination_markers == ["<TASK_COMPLETE>", "<END>", "<DONE>"]

    def test_default_system_prompt_generation(self, mock_llm_provider, subagent_ids):
        """Test that default system prompt is generated."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        system_prompt = agent.context._system_prompt
        assert system_prompt is not None
        assert "coordinator" in system_prompt.lower()
        assert "subagent" in system_prompt.lower()
        for subagent_id in subagent_ids:
            assert subagent_id in system_prompt


class TestCoordinationTools:
    """Test coordination tool functionality."""

    @pytest.mark.asyncio
    async def test_send_to_agent_tool_registration(self, mock_llm_provider, subagent_ids):
        """Test that send_to_agent tool is registered during setup."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        tool_names = [tool.name for tool in agent.tools]
        assert "send_to_agent" in tool_names

    @pytest.mark.asyncio
    async def test_list_subagents_tool_registration(self, mock_llm_provider, subagent_ids):
        """Test that list_subagents tool is registered."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        tool_names = [tool.name for tool in agent.tools]
        assert "list_subagents" in tool_names

    @pytest.mark.asyncio
    async def test_send_to_agent_tool_execution_valid(self, mock_llm_provider, subagent_ids):
        """Test send_to_agent tool sends message correctly."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session="test_session",
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        # Mock the send method
        agent.llm_behaviour.send = AsyncMock()

        response_msg = Message()
        response_msg.sender = "subagent1@localhost"
        response_msg.to = "coordinator@localhost"
        response_msg.thread = "test_session"
        response_msg.body = "test response"
        agent.llm_behaviour.receive = AsyncMock(return_value=response_msg)

        # Find and execute the send_to_agent tool
        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")
        result = await send_tool.execute(agent_id="subagent1@localhost", message="test command")

        # Verify message was sent
        assert agent.llm_behaviour.send.called
        sent_msg = agent.llm_behaviour.send.call_args[0][0]
        assert sent_msg.to == "subagent1@localhost"
        assert sent_msg.body == "test command"
        assert sent_msg.thread == "test_session"
        assert sent_msg.metadata.get("message_type") == "llm"
        assert sent_msg.metadata.get("coordination_session") == "test_session"

        # Verify status returns to idle after response handled
        assert agent.agent_status["subagent1@localhost"] == "idle"

        # Verify return message contains response information
        assert "Response from subagent1@localhost" in result
        assert "test response" in result

    @pytest.mark.asyncio
    async def test_send_to_agent_tool_invalid_agent_id(self, mock_llm_provider, subagent_ids):
        """Test error handling for unknown subagent."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        agent.llm_behaviour.send = AsyncMock()

        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")
        result = await send_tool.execute(agent_id="unknown@localhost", message="test command")

        # Should return error message
        assert "Error" in result
        assert "not a registered subagent" in result
        # Should not send message
        assert not agent.llm_behaviour.send.called

    @pytest.mark.asyncio
    async def test_list_subagents_tool_execution(self, mock_llm_provider, subagent_ids):
        """Test list_subagents tool returns correct output."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session="test_session",
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        list_tool = next(t for t in agent.tools if t.name == "list_subagents")
        result = await list_tool.execute()

        # Check output format
        assert "test_session" in result
        for subagent_id in subagent_ids:
            assert subagent_id in result
            assert "idle" in result  # Initial status

    @pytest.mark.asyncio
    async def test_list_subagents_tool_with_status_updates(self, mock_llm_provider, subagent_ids):
        """Test that list_subagents reflects status changes."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        # Update status
        agent.agent_status["subagent1@localhost"] = "working"
        agent.agent_status["subagent2@localhost"] = "timeout"

        list_tool = next(t for t in agent.tools if t.name == "list_subagents")
        result = await list_tool.execute()

        assert "subagent1@localhost" in result
        assert "working" in result
        assert "subagent2@localhost" in result
        assert "timeout" in result

    @pytest.mark.asyncio
    async def test_coordination_tools_parameter_validation(self, mock_llm_provider, subagent_ids):
        """Test tool schema parameter validation."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")
        assert "agent_id" in send_tool.parameters["required"]
        assert "message" in send_tool.parameters["required"]

        list_tool = next(t for t in agent.tools if t.name == "list_subagents")
        assert list_tool.parameters["required"] == []


class TestRoutingAndMessageFlow:
    """Test routing and message flow logic."""

    def test_default_routing_function_subagent_response(self, mock_llm_provider, subagent_ids):
        """Test that subagent responses route to coordinator."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"
        msg.to = "coordinator@localhost"

        response = "Task completed"
        context = {}

        result = agent.routing_function(msg, response, context)

        assert result == str(agent.jid)

    def test_default_routing_function_external_message(self, mock_llm_provider, subagent_ids):
        """Test that external messages route back to sender."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        msg = Mock(spec=Message)
        msg.sender = "user@localhost"
        msg.to = "coordinator@localhost"

        response = "Processing your request"
        context = {}

        result = agent.routing_function(msg, response, context)

        assert result == "user@localhost"

    def test_default_routing_function_completion_requires_complete_task(self, mock_llm_provider, subagent_ids):
        """Test response markers alone do not trigger completion routing."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        # Set original requester
        agent._original_requester = "user@localhost"

        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"

        response = "All tasks complete. <TASK_COMPLETE>"
        context = {}

        result = agent.routing_function(msg, response, context)

        assert result == str(agent.jid)

    def test_default_routing_function_original_requester_tracking(self, mock_llm_provider, subagent_ids):
        """Test that first external message sets original requester."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        assert agent._original_requester is None

        msg = Mock(spec=Message)
        msg.sender = "user@localhost"
        response = "Starting coordination"
        context = {}

        agent.routing_function(msg, response, context)

        assert agent._original_requester == "user@localhost"

        # Second external message should not override
        msg2 = Mock(spec=Message)
        msg2.sender = "other_user@localhost"
        agent.routing_function(msg2, response, context)

        assert agent._original_requester == "user@localhost"

    def test_routing_without_completion_marker(self, mock_llm_provider, subagent_ids):
        """Test no premature completion routing."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        agent._original_requester = "user@localhost"

        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"

        response = "Task in progress, no completion marker"
        context = {}

        result = agent.routing_function(msg, response, context)

        # Should route to coordinator, not original requester
        assert result == str(agent.jid)
        assert result != "user@localhost"

    def test_routing_with_complete_task_flag(self, mock_llm_provider, subagent_ids):
        """Test completion routing is driven by the complete-task flag."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        agent._original_requester = "user@localhost"

        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"
        context = {}
        agent._task_completed = True

        result = agent.routing_function(msg, "Work finished <TASK_COMPLETE>", context)

        assert result == "user@localhost"
        assert agent._task_completed is False
        assert agent._original_requester is None

    def test_routing_function_context_parameter(self, mock_llm_provider, subagent_ids):
        """Test that context dict is passed to routing function."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"
        response = "Test response"
        context = {"conversation_id": "test_conv", "state": {"interaction_count": 5}}

        # Should not raise error with context parameter
        result = agent.routing_function(msg, response, context)
        assert result == str(agent.jid)


class TestAgentStatusTracking:
    """Test agent status tracking functionality."""

    def test_agent_status_initialization(self, mock_llm_provider, subagent_ids):
        """Test that initial status is 'idle'."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        assert len(agent.agent_status) == len(subagent_ids)
        for subagent_id in subagent_ids:
            assert agent.agent_status[subagent_id] == "idle"

    @pytest.mark.asyncio
    async def test_agent_status_update_on_send(self, mock_llm_provider, subagent_ids):
        """Test status updates when command sent."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        agent.llm_behaviour.send = AsyncMock()
        response_msg = Message()
        response_msg.sender = "subagent1@localhost"
        response_msg.to = "coordinator@localhost"
        response_msg.thread = agent.coordination_session
        response_msg.body = "Acknowledged"
        agent.llm_behaviour.receive = AsyncMock(return_value=response_msg)

        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")
        await send_tool.execute(agent_id="subagent1@localhost", message="test")

        assert agent.agent_status["subagent1@localhost"] == "idle"

    @pytest.mark.asyncio
    async def test_agent_status_update_on_response(self, mock_llm_provider, subagent_ids):
        """Test status updates when response received."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        msg = Message(to="coordinator@localhost")
        msg.sender = "subagent1@localhost"
        msg.thread = agent.coordination_session
        msg.body = "Response"

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")

        agent.llm_behaviour.send = AsyncMock()
        agent.llm_behaviour.receive = AsyncMock(return_value=msg)

        await send_tool.execute(agent_id="subagent1@localhost", message="do work")

        assert agent.agent_status["subagent1@localhost"] == "idle"
        conversation = agent.context._conversations.get(agent.coordination_session, [])
        assert any(entry.get("content") == "Response" for entry in conversation)

    @pytest.mark.asyncio
    async def test_agent_status_not_updated_for_external(self, mock_llm_provider, subagent_ids):
        """Test that external messages don't affect status."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        agent.llm_behaviour.send = AsyncMock()

        external_msg = Message()
        external_msg.sender = "external_user@localhost"
        external_msg.to = "coordinator@localhost"
        external_msg.thread = agent.coordination_session
        external_msg.body = "External message"

        response_msg = Message()
        response_msg.sender = "subagent1@localhost"
        response_msg.to = "coordinator@localhost"
        response_msg.thread = agent.coordination_session
        response_msg.body = "Done"

        event = asyncio.Event()

        async def receive_side_effect(*args, **kwargs):
            if not event.is_set():
                event.set()
                return external_msg
            return response_msg

        agent.llm_behaviour.receive = AsyncMock(side_effect=receive_side_effect)

        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")

        task = asyncio.create_task(send_tool.execute(agent_id="subagent1@localhost", message="test"))

        await event.wait()
        assert agent.agent_status["subagent1@localhost"] == "working"

        await task
        assert agent.agent_status["subagent1@localhost"] == "idle"


class TestIntegration:
    """Test integration aspects of CoordinatorAgent."""

    def test_inheritance_from_llm_agent(self, mock_llm_provider, subagent_ids):
        """Test that CoordinatorAgent properly inherits from LLMAgent."""
        from spade_llm.agent.llm_agent import LLMAgent

        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        assert isinstance(agent, LLMAgent)
        assert hasattr(agent, "add_tool")
        assert hasattr(agent, "get_tools")
        assert hasattr(agent, "reset_conversation")
        assert hasattr(agent, "get_conversation_state")

    def test_context_sharing_between_coordinator_and_subagents(
        self, mock_llm_provider, subagent_ids, coordination_session_id
    ):
        """Test asymmetric visibility pattern."""
        from spade_llm.context import ContextManager

        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session=coordination_session_id,
        )

        # Add messages to coordinator's context
        msg1 = Mock(spec=Message)
        msg1.sender = "coordinator@localhost"
        msg1.to = "subagent1@localhost"
        msg1.body = "Do task 1"
        msg1.thread = coordination_session_id
        agent.context.add_message(msg1)

        msg2 = Mock(spec=Message)
        msg2.sender = "subagent1@localhost"
        msg2.to = "coordinator@localhost"
        msg2.body = "Task 1 done"
        msg2.thread = coordination_session_id
        agent.context.add_message(msg2)

        # Coordinator sees both messages
        coordinator_messages = agent.context._conversations[coordination_session_id]
        assert len(coordinator_messages) == 2

        # Simulate subagent context (isolated)
        subagent_context = ContextManager()
        subagent_msg = Mock(spec=Message)
        subagent_msg.sender = "coordinator@localhost"
        subagent_msg.to = "subagent1@localhost"
        subagent_msg.body = "Do task 1"
        subagent_msg.thread = coordination_session_id
        subagent_context.add_message(subagent_msg, coordination_session_id)

        # Subagent sees only its message
        subagent_messages = subagent_context._conversations[coordination_session_id]
        assert len(subagent_messages) == 1

    def test_multiple_subagent_coordination(self, mock_llm_provider):
        """Test coordinator managing multiple subagents."""
        many_subagents = [f"subagent{i}@localhost" for i in range(5)]

        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=many_subagents
        )

        # Simulate messages from each subagent
        for subagent_id in many_subagents:
            msg = Mock(spec=Message)
            msg.sender = subagent_id
            msg.to = "coordinator@localhost"
            msg.body = f"Message from {subagent_id}"
            msg.thread = agent.coordination_session
            agent.context.add_message(msg)

        # All messages in shared context
        messages = agent.context._conversations[agent.coordination_session]
        assert len(messages) == 5

        # All statuses tracked
        assert len(agent.agent_status) == 5
        assert all(agent_id in agent.agent_status for agent_id in many_subagents)

    def test_concurrent_coordination_sessions(self, mock_llm_provider, subagent_ids):
        """Test that multiple coordinators don't interfere."""
        agent1 = CoordinatorAgent(
            jid="coordinator1@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session="session_1",
        )

        agent2 = CoordinatorAgent(
            jid="coordinator2@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session="session_2",
        )

        # Add messages to each
        msg1 = Mock(spec=Message)
        msg1.sender = "subagent1@localhost"
        msg1.to = "coordinator1@localhost"
        msg1.body = "Session 1 message"
        msg1.thread = "session_1"
        agent1.context.add_message(msg1)

        msg2 = Mock(spec=Message)
        msg2.sender = "subagent1@localhost"
        msg2.to = "coordinator2@localhost"
        msg2.body = "Session 2 message"
        msg2.thread = "session_2"
        agent2.context.add_message(msg2)

        # Contexts remain isolated
        assert "session_1" in agent1.context._conversations
        assert "session_2" not in agent1.context._conversations
        assert "session_2" in agent2.context._conversations
        assert "session_1" not in agent2.context._conversations

    @pytest.mark.asyncio
    async def test_behaviour_configuration_propagation(self, mock_llm_provider, subagent_ids):
        """Test that configuration reaches LLMBehaviour."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        behaviour = agent.llm_behaviour

        assert behaviour.provider == mock_llm_provider
        assert behaviour.routing_function == agent.routing_function
        assert behaviour.context == agent.context
        assert behaviour.termination_markers == agent.termination_markers


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_coordinator_with_single_subagent(self, mock_llm_provider):
        """Test minimum viable coordination."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=["single_agent@localhost"],
        )

        assert len(agent.agent_status) == 1
        assert "single_agent@localhost" in agent.agent_status

    def test_coordinator_with_special_characters_in_jids(self, mock_llm_provider):
        """Test JID handling with special characters."""
        special_jids = ["agent.with.dots@localhost", "agent_with_underscores@localhost", "agent123@localhost"]

        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=special_jids
        )

        assert agent.subagent_ids == set(special_jids)

        # Test message routing
        msg = Mock(spec=Message)
        msg.sender = "agent.with.dots@localhost"
        msg.to = "coordinator@localhost"
        msg.thread = "test"

        conv_id = agent.context._get_coordination_conversation_id(msg)
        assert conv_id == agent.coordination_session

    def test_very_long_coordination_session_name(self, mock_llm_provider, subagent_ids):
        """Test long session names handled correctly."""
        long_session = "a" * 1000

        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            coordination_session=long_session,
        )

        assert agent.coordination_session == long_session

    def test_subagent_id_case_sensitivity(self, mock_llm_provider):
        """Test that exact JID matching is required."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=["Agent@localhost"],  # Capital A
        )

        msg = Mock(spec=Message)
        msg.sender = "agent@localhost"  # Lowercase a
        msg.to = "coordinator@localhost"
        msg.thread = "test"

        conv_id = agent.context._get_coordination_conversation_id(msg)

        # Should NOT be treated as subagent (case-sensitive)
        assert conv_id != agent.coordination_session

    def test_context_with_no_system_prompt(self, mock_llm_provider, subagent_ids):
        """Test coordination with explicit None system prompt."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost",
            password="password",
            provider=mock_llm_provider,
            subagent_ids=subagent_ids,
            system_prompt=None,
        )

        # Should still work, with None prompt
        assert agent.context._system_prompt is None

    @pytest.mark.asyncio
    async def test_tool_execution_with_mock_behaviour(self, mock_llm_provider, subagent_ids):
        """Test tool execution with properly mocked behaviour."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        with patch.object(agent, "add_behaviour"):
            await agent.setup()

        # Mock the send method as AsyncMock
        agent.llm_behaviour.send = AsyncMock()
        response_msg = Message()
        response_msg.sender = "subagent1@localhost"
        response_msg.to = "coordinator@localhost"
        response_msg.thread = agent.coordination_session
        response_msg.body = "test command response"
        agent.llm_behaviour.receive = AsyncMock(return_value=response_msg)

        send_tool = next(t for t in agent.tools if t.name == "send_to_agent")

        # Execute tool
        result = await send_tool.execute(agent_id="subagent1@localhost", message="test command")

        assert "Response from subagent1@localhost" in result
        assert "test command response" in result
        assert agent.llm_behaviour.send.called

    def test_routing_function_with_none_original_requester(self, mock_llm_provider, subagent_ids):
        """Test routing when original requester is None."""
        agent = CoordinatorAgent(
            jid="coordinator@localhost", password="password", provider=mock_llm_provider, subagent_ids=subagent_ids
        )

        msg = Mock(spec=Message)
        msg.sender = "subagent1@localhost"
        response = "Task done <TASK_COMPLETE>"
        context = {}

        # _original_requester is None
        assert agent._original_requester is None

        result = agent.routing_function(msg, response, context)

        # Should route to coordinator since no original requester
        assert result == str(agent.jid)
