"""Tests for RetrievalTool class."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch

from spade.message import Message
from spade_llm.tools.retrieval_tool import RetrievalTool
from spade_llm.tools.llm_tool import LLMTool


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.jid = "llm_agent@localhost"
    agent.add_behaviour = Mock()
    return agent


class TestRetrievalToolInitialization:
    """Test RetrievalTool initialization."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        assert tool.retrieval_agent_jid == "retrieval@localhost"
        assert tool.agent_instance is None
        assert tool.default_k == 4
        assert tool.timeout == 30
        assert tool.name == "retrieve_documents"
        assert tool.description is not None

    def test_init_full_parameters(self, mock_agent):
        """Test initialization with all parameters."""
        tool = RetrievalTool(
            retrieval_agent_jid="retrieval@localhost",
            agent_instance=mock_agent,
            default_k=10,
            timeout=60,
            name="custom_retrieval",
            description="Custom description"
        )

        assert tool.retrieval_agent_jid == "retrieval@localhost"
        assert tool.agent_instance == mock_agent
        assert tool.default_k == 10
        assert tool.timeout == 60
        assert tool.name == "custom_retrieval"
        assert tool.description == "Custom description"

    def test_inherits_from_llm_tool(self):
        """Test that RetrievalTool inherits from LLMTool."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        assert isinstance(tool, LLMTool)

    def test_init_default_description(self):
        """Test that default description is set correctly."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        assert "retrieve relevant documents" in tool.description.lower()

    def test_parameters_schema(self):
        """Test that parameters schema is correctly defined."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        assert tool.parameters["type"] == "object"
        assert "query" in tool.parameters["properties"]
        assert "k" in tool.parameters["properties"]
        assert "filters" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]

    def test_init_invalid_default_k_raises_error(self):
        """Test that initialization raises ValueError for invalid default_k."""
        with pytest.raises(ValueError, match="must be between 1 and 20"):
            RetrievalTool(retrieval_agent_jid="retrieval@localhost", default_k=0)
        
        with pytest.raises(ValueError, match="must be between 1 and 20"):
            RetrievalTool(retrieval_agent_jid="retrieval@localhost", default_k=21)

    def test_init_invalid_timeout_raises_error(self):
        """Test that initialization raises ValueError for non-positive timeout."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            RetrievalTool(retrieval_agent_jid="retrieval@localhost", timeout=0)
            
        with pytest.raises(ValueError, match="must be a positive integer"):
            RetrievalTool(retrieval_agent_jid="retrieval@localhost", timeout=-10)


class TestRetrievalToolSetAgent:
    """Test set_agent method."""

    def test_set_agent(self, mock_agent):
        """Test setting the agent instance."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        assert tool.agent_instance == mock_agent

    def test_set_agent_logging(self, mock_agent):
        """Test that set_agent logs the binding."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        with patch("spade_llm.tools.retrieval_tool.logger") as mock_logger:
            tool.set_agent(mock_agent)
            mock_logger.info.assert_called()


class TestRetrievalToolRetrieveDocuments:
    """Test _retrieve_documents method."""

    def setup_method(self):
        """Set up mocks for each test in this class."""
        self.mock_response = Mock(spec=Message)
        self.mock_response.body = json.dumps({"documents": []})
        self.async_mock_sender = AsyncMock(return_value=self.mock_response)

    @pytest.mark.asyncio
    async def test_retrieve_without_agent_instance(self):
        """Test that retrieval fails gracefully without agent instance."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        result = await tool._retrieve_documents("test query")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data
        assert "not properly initialized" in data["error"]

    @pytest.mark.asyncio
    async def test_retrieve_basic_query(self, mock_agent):
        """Test basic document retrieval."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        # Mock the response with specific documents
        custom_response = Mock(spec=Message)
        custom_response.body = json.dumps({
            "documents": [
                {"content": "Doc 1", "metadata": {}, "rank": 1},
                {"content": "Doc 2", "metadata": {}, "rank": 2}
            ]
        })
        
        tool._send_and_wait_for_response = AsyncMock(return_value=custom_response)

        result = await tool._retrieve_documents("test query")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "documents" in data
        assert len(data["documents"]) == 2

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_k(self, mock_agent):
        """Test retrieval with custom k parameter."""
        tool = RetrievalTool(
            retrieval_agent_jid="retrieval@localhost",
            default_k=4
        )
        tool.set_agent(mock_agent)
        tool._send_and_wait_for_response = AsyncMock(return_value=self.mock_response)

        await tool._retrieve_documents("test query", k=10)

        # Verify the sent message had k=10
        call_args = tool._send_and_wait_for_response.call_args[0][0]
        sent_data = json.loads(call_args.body)
        assert sent_data["k"] == 10

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, mock_agent):
        """Test retrieval with metadata filters."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)
        tool._send_and_wait_for_response = AsyncMock(return_value=self.mock_response)

        filters = {"type": "research", "year": 2024}
        await tool._retrieve_documents("test query", filters=filters)

        # Verify filters were sent
        call_args = tool._send_and_wait_for_response.call_args[0][0]
        sent_data = json.loads(call_args.body)
        assert "filters" in sent_data
        assert sent_data["filters"] == filters

    @pytest.mark.asyncio
    async def test_retrieve_timeout_handling(self, mock_agent):
        """Test handling of retrieval timeout."""
        tool = RetrievalTool(
            retrieval_agent_jid="retrieval@localhost",
            timeout=30
        )
        tool.set_agent(mock_agent)

        # Simulate timeout (no response)
        tool._send_and_wait_for_response = AsyncMock(return_value=None)

        result = await tool._retrieve_documents("test query")

        data = json.loads(result)
        assert "error" in data
        assert "timeout" in data["error"].lower()
        assert data["query"] == "test query"

    @pytest.mark.asyncio
    async def test_retrieve_exception_handling(self, mock_agent):
        """Test handling of exceptions during retrieval."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        # Simulate exception
        tool._send_and_wait_for_response = AsyncMock(
            side_effect=Exception("Connection error")
        )

        result = await tool._retrieve_documents("test query")

        data = json.loads(result)
        assert "error" in data
        assert "Connection error" in data["error"]

    @pytest.mark.asyncio
    async def test_retrieve_message_structure(self, mock_agent):
        """Test that query message has correct structure."""
        tool = RetrievalTool(
            retrieval_agent_jid="retrieval@localhost"
        )
        tool.set_agent(mock_agent)
        tool._send_and_wait_for_response = AsyncMock(return_value=self.mock_response)

        await tool._retrieve_documents("test query", k=5)

        # Verify message structure
        sent_msg = tool._send_and_wait_for_response.call_args[0][0]
        assert sent_msg.to == "retrieval@localhost"
        
        sent_data = json.loads(sent_msg.body)
        assert sent_data["query"] == "test query"
        assert sent_data["k"] == 5
        # include_scores is no longer sent as it's not supported by retrievers


class TestRetrievalToolFormatResponse:
    """Test _format_response method."""

    def test_format_response_with_documents(self):
        """Test formatting response with documents."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        response_body = json.dumps({
            "documents": [
                {"content": "Doc 1", "metadata": {"id": 1}},
                {"content": "Doc 2", "metadata": {"id": 2}}
            ]
        })

        result = tool._format_response(response_body)
        data = json.loads(result)

        assert "documents" in data
        assert len(data["documents"]) == 2
        # Check that ranks were added
        assert data["documents"][0]["rank"] == 1
        assert data["documents"][1]["rank"] == 2

    def test_format_response_with_error(self):
        """Test formatting response containing error."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        response_body = json.dumps({"error": "Retrieval failed"})

        result = tool._format_response(response_body)
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "Retrieval failed"

    def test_format_response_empty_documents(self):
        """Test formatting response with no documents."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        response_body = json.dumps({"documents": []})

        result = tool._format_response(response_body)
        data = json.loads(result)

        assert "message" in data
        assert "no documents found" in data["message"].lower()

    def test_format_response_invalid_json(self):
        """Test formatting invalid JSON response."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        result = tool._format_response("invalid json")
        data = json.loads(result)

        assert "error" in data
        assert "Failed to parse" in data["error"]
        assert "raw_response" in data

    def test_format_response_adds_ranks(self):
        """Test that ranks are added to documents."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        response_body = json.dumps({
            "documents": [
                {"content": "First"},
                {"content": "Second"},
                {"content": "Third"}
            ]
        })

        result = tool._format_response(response_body)
        data = json.loads(result)

        assert data["documents"][0]["rank"] == 1
        assert data["documents"][1]["rank"] == 2
        assert data["documents"][2]["rank"] == 3


class TestRetrievalToolSendAndWaitForResponse:
    """Test _send_and_wait_for_response method."""

    @pytest.mark.asyncio
    async def test_send_and_wait_without_agent_instance(self):
        """Test that method raises error without agent instance."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        
        msg = Mock(spec=Message)
        msg.to = "retrieval@localhost"
        msg.body = json.dumps({"query": "test"})

        with pytest.raises(ValueError, match="Agent instance is not set"):
            await tool._send_and_wait_for_response(msg)

    @pytest.mark.asyncio
    async def test_send_and_wait_successful_response(self, mock_agent):
        """Test successful send and receive by controlling the behaviour."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost", timeout=5)
        tool.set_agent(mock_agent)

        msg = Mock(spec=Message)
        response_msg = Mock(spec=Message)

        # This is the key: we intercept the behaviour that the method creates
        # and control its outcome.
        def setup_behaviour_response(behaviour, template):
            # Simulate the behaviour running and receiving a message
            behaviour.response = response_msg
            # We need an awaitable join method
            behaviour.join = AsyncMock()

        mock_agent.add_behaviour = Mock(side_effect=setup_behaviour_response)

        # Now, call the REAL method
        result = await tool._send_and_wait_for_response(msg)

        # Assertions
        mock_agent.add_behaviour.assert_called_once()
        assert result is response_msg

    @pytest.mark.asyncio
    async def test_send_and_wait_timeout(self, mock_agent):
        """Test timeout scenario by controlling the behaviour."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost", timeout=1)
        tool.set_agent(mock_agent)

        msg = Mock(spec=Message)

        # Similar to the success case, but we simulate a timeout
        # by leaving the behaviour's response as None.
        def setup_behaviour_timeout(behaviour, template):
            behaviour.response = None  # No response was received
            behaviour.join = AsyncMock()

        mock_agent.add_behaviour = Mock(side_effect=setup_behaviour_timeout)

        # Call the REAL method
        result = await tool._send_and_wait_for_response(msg)

        # Assertions
        mock_agent.add_behaviour.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_send_and_wait_creates_behaviour(self, mock_agent):
        """Test that behaviour is created and added to agent."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost", timeout=10)
        tool.set_agent(mock_agent)

        msg = Mock(spec=Message)
        msg.to = "retrieval@localhost"
        msg.body = json.dumps({"query": "test"})

        # Create a real mock for add_behaviour to track calls
        add_behaviour_calls = []
        
        def track_add_behaviour(behaviour, template):
            add_behaviour_calls.append((behaviour, template))
            # Mock the join to return immediately
            behaviour.join = AsyncMock()
            behaviour.response = None
        
        mock_agent.add_behaviour = track_add_behaviour

        await tool._send_and_wait_for_response(msg)

        # Verify add_behaviour was called
        assert len(add_behaviour_calls) == 1
        behaviour, template = add_behaviour_calls[0]
        
        # Verify template configuration
        assert template.sender == "retrieval@localhost"
        assert template.metadata.get("message_type") == "retrieval_response"

    @pytest.mark.asyncio
    async def test_send_and_wait_uses_correct_timeout(self, mock_agent):
        """Test that the correct timeout value is used."""
        custom_timeout = 45
        tool = RetrievalTool(
            retrieval_agent_jid="retrieval@localhost",
            timeout=custom_timeout
        )
        tool.set_agent(mock_agent)

        msg = Mock(spec=Message)
        msg.to = "retrieval@localhost"
        msg.body = json.dumps({"query": "test"})

        # Track the behaviour creation
        created_behaviours = []
        
        def track_add_behaviour(behaviour, template):
            created_behaviours.append(behaviour)
            behaviour.join = AsyncMock()
            behaviour.response = None
        
        mock_agent.add_behaviour = track_add_behaviour

        await tool._send_and_wait_for_response(msg)

        # Verify behaviour was created with correct timeout
        assert len(created_behaviours) == 1
        assert created_behaviours[0].response_timeout == custom_timeout

    @pytest.mark.asyncio
    async def test_send_and_wait_message_sent_in_behaviour(self, mock_agent):
        """Test that message is sent within the behaviour."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        msg = Mock(spec=Message)
        msg.to = "retrieval@localhost"
        msg.body = json.dumps({"query": "test"})

        # Track behaviour creation and execution
        behaviour_instance = None
        
        def track_add_behaviour(behaviour, template):
            nonlocal behaviour_instance
            behaviour_instance = behaviour
            behaviour.join = AsyncMock()
            behaviour.response = None
        
        mock_agent.add_behaviour = track_add_behaviour

        await tool._send_and_wait_for_response(msg)

        # Verify the behaviour has the message
        assert behaviour_instance is not None
        assert behaviour_instance.query_msg == msg


class TestRetrievalToolIntegration:
    """Integration tests for RetrievalTool."""

    def test_tool_registration_with_agent(self, mock_agent):
        """Test that tool can be registered with an agent."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        assert tool.agent_instance == mock_agent

    def test_tool_parameter_validation(self):
        """Test that tool parameters are validated correctly."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        # Verify k parameter constraints
        k_param = tool.parameters["properties"]["k"]
        assert k_param["minimum"] == 1
        assert k_param["maximum"] == 20
        assert k_param["type"] == "integer"

    def test_retrieval_response_behaviour_is_nested(self):
        """Test that _WaitForResponseBehaviour is nested inside RetrievalTool."""
        from spade_llm.tools.retrieval_tool import RetrievalTool
        from spade.behaviour import OneShotBehaviour
        
        # Verify it's a nested class
        assert hasattr(RetrievalTool, '_WaitForResponseBehaviour')
        nested_class = RetrievalTool._WaitForResponseBehaviour
        assert isinstance(nested_class, type)
        assert issubclass(nested_class, OneShotBehaviour)


class TestRetrievalToolEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("invalid_query", ["", "   ", None])
    @pytest.mark.asyncio
    async def test_retrieve_invalid_query(self, mock_agent, invalid_query):
        """Test retrieval with invalid queries (empty, whitespace-only, None)."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        result = await tool._retrieve_documents(invalid_query)

        # Should return an error for invalid query
        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data
        assert "cannot be empty" in data["error"].lower() or "whitespace" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_retrieve_very_long_query(self, mock_agent):
        """Test retrieval with very long query."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        mock_response = Mock(spec=Message)
        mock_response.body = json.dumps({"documents": []})
        tool._send_and_wait_for_response = AsyncMock(return_value=mock_response)

        long_query = "test " * 1000
        result = await tool._retrieve_documents(long_query)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_retrieve_with_unicode_query(self, mock_agent):
        """Test retrieval with unicode characters."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        mock_response = Mock(spec=Message)
        mock_response.body = json.dumps({"documents": []})
        tool._send_and_wait_for_response = AsyncMock(return_value=mock_response)

        unicode_query = "Tëst quëry with 中文 and émojis 🔍"
        result = await tool._retrieve_documents(unicode_query)

        assert isinstance(result, str)

    @pytest.mark.parametrize("invalid_k", [0, -1, 21, 101])
    @pytest.mark.asyncio
    async def test_retrieve_with_invalid_k_returns_error(self, mock_agent, invalid_k):
        """Test that retrieval with k outside 1-20 range returns an error."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        # This call should not reach the agent
        tool._send_and_wait_for_response = AsyncMock()

        result = await tool._retrieve_documents("test query", k=invalid_k)

        data = json.loads(result)
        assert "error" in data
        assert "must be between 1 and 20" in data["error"]
        tool._send_and_wait_for_response.assert_not_called()

    @pytest.mark.parametrize("valid_k", [1, 10, 20])
    @pytest.mark.asyncio
    async def test_retrieve_k_valid_boundary_values(self, mock_agent, valid_k):
        """Test retrieval with k values at valid boundaries (1, 20)."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        mock_response = Mock(spec=Message)
        mock_response.body = json.dumps({"documents": []})
        tool._send_and_wait_for_response = AsyncMock(return_value=mock_response)

        result = await tool._retrieve_documents("test", k=valid_k)

        # Should process without error
        assert isinstance(result, str)
        
        # Verify the k value was sent to the retrieval agent
        call_args = tool._send_and_wait_for_response.call_args[0][0]
        sent_data = json.loads(call_args.body)
        assert sent_data["k"] == valid_k

    @pytest.mark.asyncio
    async def test_retrieve_none_response_body(self, mock_agent):
        """Test handling of None response body."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        mock_response = Mock(spec=Message)
        mock_response.body = None
        tool._send_and_wait_for_response = AsyncMock(return_value=mock_response)

        result = await tool._retrieve_documents("test")

        # Should handle gracefully
        data = json.loads(result)
        assert "error" in data or "message" in data

    def test_format_response_with_special_characters(self):
        """Test formatting response with special characters."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        response_body = json.dumps({
            "documents": [
                {"content": 'Document with "quotes" and \n newlines', "metadata": {}}
            ]
        })

        result = tool._format_response(response_body)
        data = json.loads(result)

        assert "documents" in data
        assert 'quotes' in data["documents"][0]["content"]

    def test_init_with_empty_jid(self):
        """Test initialization with empty JID."""
        tool = RetrievalTool(retrieval_agent_jid="")
        assert tool.retrieval_agent_jid == ""


class TestRetrievalToolLogging:
    """Test logging behavior of RetrievalTool."""

    @pytest.mark.asyncio
    async def test_retrieve_logs_query(self, mock_agent):
        """Test that query and JID are logged."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        mock_response = Mock(spec=Message)
        mock_response.body = json.dumps({"documents": []})
        tool._send_and_wait_for_response = AsyncMock(return_value=mock_response)

        with patch("spade_llm.tools.retrieval_tool.logger") as mock_logger:
            await tool._retrieve_documents("test query")
            mock_logger.info.assert_called()
            
            # Verify specific content in log message
            log_call_args = mock_logger.info.call_args_list
            log_messages = [str(call[0][0]) for call in log_call_args]
            assert any("retrieval@localhost" in msg and "test query" in msg for msg in log_messages), \
                "Expected log to contain both JID and query"

    @pytest.mark.asyncio
    async def test_retrieve_logs_timeout(self, mock_agent):
        """Test that timeout is logged with specific details."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost", timeout=30)
        tool.set_agent(mock_agent)

        tool._send_and_wait_for_response = AsyncMock(return_value=None)

        with patch("spade_llm.tools.retrieval_tool.logger") as mock_logger:
            await tool._retrieve_documents("test")
            mock_logger.warning.assert_called()
            
            # Verify timeout details in log message
            warning_msg = str(mock_logger.warning.call_args[0][0])
            assert "timeout" in warning_msg.lower(), "Expected 'timeout' in warning message"
            assert "30" in warning_msg or "30s" in warning_msg, "Expected timeout value in warning message"

    @pytest.mark.asyncio
    async def test_retrieve_logs_errors(self, mock_agent):
        """Test that errors are logged with exception details."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")
        tool.set_agent(mock_agent)

        tool._send_and_wait_for_response = AsyncMock(side_effect=Exception("Connection error"))

        with patch("spade_llm.tools.retrieval_tool.logger") as mock_logger:
            await tool._retrieve_documents("test")
            mock_logger.error.assert_called()
            
            # Verify error details in log message
            error_msg = str(mock_logger.error.call_args[0][0])
            assert "error" in error_msg.lower(), "Expected 'error' in error message"
            assert "Connection error" in error_msg, "Expected exception message in error log"

    def test_format_response_logs_parse_error(self):
        """Test that JSON parse errors are logged with details."""
        tool = RetrievalTool(retrieval_agent_jid="retrieval@localhost")

        with patch("spade_llm.tools.retrieval_tool.logger") as mock_logger:
            tool._format_response("invalid json")
            mock_logger.error.assert_called()
            
            # Verify parse error details in log message
            error_msg = str(mock_logger.error.call_args[0][0])
            assert "parse" in error_msg.lower() or "json" in error_msg.lower(), \
                "Expected 'parse' or 'json' in error message"
