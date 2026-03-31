"""Tests for HumanInteractionBehaviour class."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from spade.behaviour import OneShotBehaviour
from spade.message import Message

from spade_llm.behaviour.human_interaction import HumanInteractionBehaviour


class TestHumanInteractionBehaviourInitialization:
    """Test HumanInteractionBehaviour initialization."""

    def test_init_minimal_parameters(self):
        """Test initialization with minimal parameters."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="What is the answer?")

        assert behaviour.human_jid == "expert@localhost"
        assert behaviour.question == "What is the answer?"
        assert behaviour.context is None
        assert behaviour.timeout == 300.0
        assert behaviour.response is None
        assert behaviour.query_id is not None
        assert len(behaviour.query_id) == 8
        assert isinstance(behaviour, OneShotBehaviour)

    def test_init_full_parameters(self):
        """Test initialization with all parameters."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost", question="What is the answer?", context="This is about testing", timeout=60.0
        )

        assert behaviour.human_jid == "expert@localhost"
        assert behaviour.question == "What is the answer?"
        assert behaviour.context == "This is about testing"
        assert behaviour.timeout == 60.0
        assert behaviour.response is None
        assert behaviour.query_id is not None
        assert len(behaviour.query_id) == 8

    def test_init_default_timeout(self):
        """Test that default timeout is 300 seconds."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")

        assert behaviour.timeout == 300.0

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question", timeout=120.0)

        assert behaviour.timeout == 120.0

    def test_init_generates_unique_query_ids(self):
        """Test that each instance generates a unique query ID."""
        behaviour1 = HumanInteractionBehaviour(human_jid="expert@localhost", question="Question 1")
        behaviour2 = HumanInteractionBehaviour(human_jid="expert@localhost", question="Question 2")

        assert behaviour1.query_id != behaviour2.query_id
        assert len(behaviour1.query_id) == 8
        assert len(behaviour2.query_id) == 8
        assert behaviour1.query_id.replace("-", "").isalnum()
        assert behaviour2.query_id.replace("-", "").isalnum()

    def test_init_inherits_from_one_shot_behaviour(self):
        """Test that HumanInteractionBehaviour inherits from OneShotBehaviour."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")

        assert isinstance(behaviour, OneShotBehaviour)

    def test_init_with_empty_strings(self):
        """Test initialization with empty strings."""
        behaviour = HumanInteractionBehaviour(human_jid="", question="", context="")

        assert behaviour.human_jid == ""
        assert behaviour.question == ""
        assert behaviour.context == ""

    def test_init_with_none_context(self):
        """Test initialization with None context (default)."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question", context=None)

        assert behaviour.context is None


class TestHumanInteractionBehaviourFormatQuestion:
    """Test the _format_question method."""

    def test_format_question_minimal(self):
        """Test formatting question without context."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="What is the answer?")

        formatted = behaviour._format_question()

        assert f"[Query {behaviour.query_id}] What is the answer?" in formatted
        assert "Context:" not in formatted
        assert "(Please reply to this message to provide your answer)" in formatted

    def test_format_question_with_context(self):
        """Test formatting question with context."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost", question="What is the answer?", context="This is about testing"
        )

        formatted = behaviour._format_question()

        assert f"[Query {behaviour.query_id}] What is the answer?" in formatted
        assert "Context: This is about testing" in formatted
        assert "(Please reply to this message to provide your answer)" in formatted

    def test_format_question_with_empty_context(self):
        """Test formatting question with empty context."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="What is the answer?", context="")

        formatted = behaviour._format_question()

        # Empty context should not add the context section
        assert f"[Query {behaviour.query_id}] What is the answer?" in formatted
        assert "Context:" not in formatted
        assert "(Please reply to this message to provide your answer)" in formatted

    def test_format_question_with_multiline_context(self):
        """Test formatting question with multiline context."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost",
            question="What is the answer?",
            context="This is line 1\nThis is line 2\nThis is line 3",
        )

        formatted = behaviour._format_question()

        assert f"[Query {behaviour.query_id}] What is the answer?" in formatted
        assert "Context: This is line 1\nThis is line 2\nThis is line 3" in formatted
        assert "(Please reply to this message to provide your answer)" in formatted

    def test_format_question_with_special_characters(self):
        """Test formatting question with special characters."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost",
            question="What is the answer? & how do you know @user?",
            context="Context with <html> tags & special chars: !@#$%^&*()",
        )

        formatted = behaviour._format_question()

        assert f"[Query {behaviour.query_id}] What is the answer? & how do you know @user?" in formatted
        assert "Context: Context with <html> tags & special chars: !@#$%^&*()" in formatted
        assert "(Please reply to this message to provide your answer)" in formatted

    def test_format_question_structure(self):
        """Test the overall structure of formatted question."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost", question="Test question", context="Test context"
        )

        formatted = behaviour._format_question()
        lines = formatted.split("\n")

        # Check overall structure
        assert len(lines) >= 4  # Query line, empty line, context line, empty line, instruction line
        assert lines[0].startswith(f"[Query {behaviour.query_id}]")
        assert "Context:" in lines[2]
        assert "(Please reply to this message to provide your answer)" in lines[-1]


class TestHumanInteractionBehaviourRun:
    """Test the run method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with XMPP client."""
        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        agent.connected_event.wait = AsyncMock()
        return agent

    @pytest.fixture
    def mock_behaviour(self, mock_agent):
        """Create a behaviour with mocked dependencies."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        behaviour.agent = mock_agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock()
        return behaviour

    @pytest.mark.asyncio
    async def test_run_successful_flow(self, mock_behaviour):
        """Test successful run flow with response."""
        # Mock response message
        response_msg = Mock()
        response_msg.body = "Here is the answer"
        mock_behaviour.receive.return_value = response_msg

        await mock_behaviour.run()

        # Verify agent connection was waited for
        mock_behaviour.agent.connected_event.wait.assert_called_once()

        # Verify message was sent
        mock_behaviour.send.assert_called_once()
        sent_msg = mock_behaviour.send.call_args[0][0]
        assert sent_msg.to == "expert@localhost"
        assert f"[Query {mock_behaviour.query_id}] Test question" in sent_msg.body
        assert sent_msg.thread == mock_behaviour.query_id
        assert sent_msg.get_metadata("type") == "human_query"
        assert sent_msg.get_metadata("query_id") == mock_behaviour.query_id

        # Verify response was received
        mock_behaviour.receive.assert_called_once_with(timeout=300.0)
        assert mock_behaviour.response == "Here is the answer"

    @pytest.mark.asyncio
    async def test_run_with_custom_timeout(self, mock_agent):
        """Test run with custom timeout."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question", timeout=60.0)
        behaviour.agent = mock_agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Verify timeout was used
        behaviour.receive.assert_called_once_with(timeout=60.0)

    @pytest.mark.asyncio
    async def test_run_no_response(self, mock_behaviour):
        """Test run when no response is received."""
        mock_behaviour.receive.return_value = None

        await mock_behaviour.run()

        # Verify message was sent
        mock_behaviour.send.assert_called_once()

        # Verify no response was set
        assert mock_behaviour.response is None

    @pytest.mark.asyncio
    async def test_run_with_context(self, mock_agent):
        """Test run with context included."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost", question="Test question", context="Test context"
        )
        behaviour.agent = mock_agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Verify message contains context
        sent_msg = behaviour.send.call_args[0][0]
        assert "Context: Test context" in sent_msg.body

    @pytest.mark.asyncio
    async def test_run_without_connected_event(self, mock_agent):
        """Test run when agent doesn't have connected_event."""
        mock_agent.connected_event = None
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        behaviour.agent = mock_agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Should still work without connected_event
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_connected_event_wait_fails(self, mock_agent):
        """Test run when waiting for connected_event fails."""
        mock_agent.connected_event.wait.side_effect = Exception("Connection error")

        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        behaviour.agent = mock_agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        # Should not raise exception
        await behaviour.run()

        # Should still send message
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_no_agent(self):
        """Test run when agent is not set."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        behaviour.agent = None
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock()

        await behaviour.run()

        # Should not send or receive anything
        behaviour.send.assert_not_called()
        behaviour.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_no_client(self, mock_agent):
        """Test run when agent has no client."""
        mock_agent.client = None

        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        behaviour.agent = mock_agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock()

        await behaviour.run()

        # Should not send or receive anything
        behaviour.send.assert_not_called()
        behaviour.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_send_exception(self, mock_behaviour):
        """Test run when send raises exception."""
        mock_behaviour.send.side_effect = Exception("Send error")
        mock_behaviour.receive.return_value = None

        # Should not raise exception
        await mock_behaviour.run()

        # Should still try to receive
        mock_behaviour.receive.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_message_metadata(self, mock_behaviour):
        """Test that message metadata is set correctly."""
        mock_behaviour.receive.return_value = None

        await mock_behaviour.run()

        # Check message metadata
        sent_msg = mock_behaviour.send.call_args[0][0]
        assert sent_msg.get_metadata("type") == "human_query"
        assert sent_msg.get_metadata("query_id") == mock_behaviour.query_id
        assert sent_msg.thread == mock_behaviour.query_id

    @pytest.mark.asyncio
    async def test_run_message_structure(self, mock_behaviour):
        """Test that message structure is correct."""
        mock_behaviour.receive.return_value = None

        await mock_behaviour.run()

        # Check message structure
        sent_msg = mock_behaviour.send.call_args[0][0]
        assert isinstance(sent_msg, Message)
        assert sent_msg.to == "expert@localhost"
        assert sent_msg.body is not None
        assert sent_msg.thread == mock_behaviour.query_id

    @pytest.mark.asyncio
    async def test_run_response_handling(self, mock_behaviour):
        """Test response handling."""
        # Test with response
        response_msg = Mock()
        response_msg.body = "Test response"
        mock_behaviour.receive.return_value = response_msg

        await mock_behaviour.run()

        assert mock_behaviour.response == "Test response"

        # Test without response
        mock_behaviour.receive.return_value = None
        mock_behaviour.response = None

        await mock_behaviour.run()

        assert mock_behaviour.response is None

    @pytest.mark.asyncio
    async def test_run_with_empty_response(self, mock_behaviour):
        """Test run with empty response body."""
        response_msg = Mock()
        response_msg.body = ""
        mock_behaviour.receive.return_value = response_msg

        await mock_behaviour.run()

        assert mock_behaviour.response == ""


class TestHumanInteractionBehaviourLogging:
    """Test logging functionality."""

    @pytest.fixture
    def mock_behaviour_with_logging(self):
        """Create behaviour with logging mocks."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        agent.connected_event.wait = AsyncMock()
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock()
        return behaviour

    @pytest.mark.asyncio
    async def test_logging_connection_error(self, mock_behaviour_with_logging):
        """Test logging when connection wait fails."""
        mock_behaviour_with_logging.agent.connected_event.wait.side_effect = Exception("Connection error")
        mock_behaviour_with_logging.receive.return_value = None

        with patch("spade_llm.behaviour.human_interaction.logger") as mock_logger:
            await mock_behaviour_with_logging.run()

            mock_logger.warning.assert_any_call("Could not wait for agent connection: Connection error")

    @pytest.mark.asyncio
    async def test_logging_no_client(self):
        """Test logging when agent has no client."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")
        agent = Mock()
        agent.client = None
        agent.connected_event = AsyncMock()
        behaviour.agent = agent

        with patch("spade_llm.behaviour.human_interaction.logger") as mock_logger:
            await behaviour.run()

            mock_logger.error.assert_any_call(f"Agent XMPP client not available for query {behaviour.query_id}")

    @pytest.mark.asyncio
    async def test_logging_send_error(self, mock_behaviour_with_logging):
        """Test logging when send fails."""
        mock_behaviour_with_logging.send.side_effect = Exception("Send error")
        mock_behaviour_with_logging.receive.return_value = None

        with patch("spade_llm.behaviour.human_interaction.logger") as mock_logger:
            await mock_behaviour_with_logging.run()

            mock_logger.warning.assert_any_call("Send error (message likely delivered): Send error")

    @pytest.mark.asyncio
    async def test_logging_no_response(self, mock_behaviour_with_logging):
        """Test logging when no response is received."""
        mock_behaviour_with_logging.receive.return_value = None

        with patch("spade_llm.behaviour.human_interaction.logger") as mock_logger:
            await mock_behaviour_with_logging.run()

            mock_logger.warning.assert_any_call(
                f"No response received for query {mock_behaviour_with_logging.query_id}"
            )

    @pytest.mark.asyncio
    async def test_logging_response_truncation(self, mock_behaviour_with_logging):
        """Test that long responses are truncated in logs."""
        long_response = "a" * 100
        response_msg = Mock()
        response_msg.body = long_response
        mock_behaviour_with_logging.receive.return_value = response_msg

        with patch("spade_llm.behaviour.human_interaction.logger") as mock_logger:
            await mock_behaviour_with_logging.run()

            # Check that response is truncated to 50 characters
            expected_log = f"Received response for query {mock_behaviour_with_logging.query_id}: " + "a" * 50 + "..."
            mock_logger.info.assert_any_call(expected_log)


class TestHumanInteractionBehaviourEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_long_question(self):
        """Test with very long question."""
        long_question = "What is the answer to " + "x" * 1000 + "?"
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question=long_question)

        formatted = behaviour._format_question()
        assert long_question in formatted
        assert f"[Query {behaviour.query_id}]" in formatted

    @pytest.mark.asyncio
    async def test_unicode_characters(self):
        """Test with unicode characters."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost",
            question="What is the answer? 你好世界 🌍",
            context="Context with émojis: 🚀 and accénts",
        )

        formatted = behaviour._format_question()
        assert "What is the answer? 你好世界 🌍" in formatted
        assert "Context with émojis: 🚀 and accénts" in formatted

    @pytest.mark.asyncio
    async def test_newlines_in_question(self):
        """Test with newlines in question."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost", question="What is the answer?\nThis is line 2\nThis is line 3"
        )

        formatted = behaviour._format_question()
        assert "What is the answer?\nThis is line 2\nThis is line 3" in formatted

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """Test with zero timeout."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question", timeout=0.0)

        assert behaviour.timeout == 0.0

    @pytest.mark.asyncio
    async def test_negative_timeout(self):
        """Test with negative timeout."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question", timeout=-10.0)

        assert behaviour.timeout == -10.0

    @pytest.mark.asyncio
    async def test_very_large_timeout(self):
        """Test with very large timeout."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question", timeout=999999.0)

        assert behaviour.timeout == 999999.0

    @pytest.mark.asyncio
    async def test_empty_human_jid(self):
        """Test with empty human JID."""
        behaviour = HumanInteractionBehaviour(human_jid="", question="Test question")

        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Should still send message to empty JID
        sent_msg = behaviour.send.call_args[0][0]
        assert sent_msg.to == ""

    @pytest.mark.asyncio
    async def test_malformed_jid(self):
        """Test with malformed JID."""
        behaviour = HumanInteractionBehaviour(human_jid="not-a-valid-jid", question="Test question")

        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Should still send message (validation is not our responsibility)
        sent_msg = behaviour.send.call_args[0][0]
        assert sent_msg.to == "not-a-valid-jid"

    @pytest.mark.asyncio
    async def test_concurrent_behaviour_instances(self):
        """Test multiple behaviour instances running concurrently."""
        behaviours = []
        agents = []

        for i in range(3):
            behaviour = HumanInteractionBehaviour(human_jid=f"expert{i}@localhost", question=f"Question {i}")
            agent = Mock()
            agent.client = Mock()
            agent.connected_event = AsyncMock()
            behaviour.agent = agent
            behaviour.send = AsyncMock()
            behaviour.receive = AsyncMock(return_value=None)

            behaviours.append(behaviour)
            agents.append(agent)

        # Run all behaviours concurrently
        await asyncio.gather(*[behaviour.run() for behaviour in behaviours])

        # Verify all sent messages
        for i, behaviour in enumerate(behaviours):
            behaviour.send.assert_called_once()
            sent_msg = behaviour.send.call_args[0][0]
            assert sent_msg.to == f"expert{i}@localhost"
            assert f"Question {i}" in sent_msg.body

    @pytest.mark.asyncio
    async def test_agent_without_connected_event_attribute(self):
        """Test agent that doesn't have connected_event attribute at all."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")

        agent = Mock()
        agent.client = Mock()
        # Don't set connected_event attribute at all
        del agent.connected_event
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        # Should not raise exception
        await behaviour.run()

        # Should still send message
        behaviour.send.assert_called_once()


class TestHumanInteractionBehaviourIntegration:
    """Integration tests for HumanInteractionBehaviour."""

    @pytest.mark.asyncio
    async def test_full_interaction_flow(self):
        """Test complete interaction flow from start to finish."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost", question="What is 2+2?", context="Simple math question", timeout=30.0
        )

        # Mock agent
        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        behaviour.agent = agent

        # Mock send and receive
        behaviour.send = AsyncMock()
        response_msg = Mock()
        response_msg.body = "The answer is 4"
        behaviour.receive = AsyncMock(return_value=response_msg)

        # Execute
        await behaviour.run()

        # Verify complete flow
        agent.connected_event.wait.assert_called_once()
        behaviour.send.assert_called_once()
        behaviour.receive.assert_called_once_with(timeout=30.0)

        # Verify message content
        sent_msg = behaviour.send.call_args[0][0]
        assert sent_msg.to == "expert@localhost"
        assert "What is 2+2?" in sent_msg.body
        assert "Context: Simple math question" in sent_msg.body
        assert sent_msg.thread == behaviour.query_id
        assert sent_msg.get_metadata("type") == "human_query"

        # Verify response
        assert behaviour.response == "The answer is 4"

    @pytest.mark.asyncio
    async def test_timeout_scenario(self):
        """Test timeout scenario."""
        behaviour = HumanInteractionBehaviour(
            human_jid="expert@localhost",
            question="What is the answer?",
            timeout=0.1,  # Very short timeout
        )

        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)  # Simulate timeout

        await behaviour.run()

        # Should handle timeout gracefully
        assert behaviour.response is None
        behaviour.receive.assert_called_once_with(timeout=0.1)

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery scenarios."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")

        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        agent.connected_event.wait.side_effect = Exception("Connection error")
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.send.side_effect = Exception("Send error")
        behaviour.receive = AsyncMock(return_value=None)

        # Should not raise exception despite multiple errors
        await behaviour.run()

        # Should still attempt to receive
        behaviour.receive.assert_called_once()
        assert behaviour.response is None

    @pytest.mark.asyncio
    async def test_query_id_consistency(self):
        """Test that query_id is consistent throughout the flow."""
        behaviour = HumanInteractionBehaviour(human_jid="expert@localhost", question="Test question")

        initial_query_id = behaviour.query_id

        agent = Mock()
        agent.client = Mock()
        agent.connected_event = AsyncMock()
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Query ID should remain the same
        assert behaviour.query_id == initial_query_id

        # Message should use the same query ID
        sent_msg = behaviour.send.call_args[0][0]
        assert sent_msg.thread == initial_query_id
        assert sent_msg.get_metadata("query_id") == initial_query_id
        assert f"[Query {initial_query_id}]" in sent_msg.body
