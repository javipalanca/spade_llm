"""Tests for RetrievalBehaviour class."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, call
import time

from spade.message import Message
from spade_llm.behaviour.retrieval_behaviour import RetrievalBehaviour
from spade_llm.rag.retrievers.base import BaseRetriever
from spade_llm.rag.core.document import Document


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    def __init__(self, documents=None, should_error=False):
        self.documents = documents or []
        self.retrieve_calls = []
        self.should_error = should_error

    async def retrieve(self, query: str, k: int = 4, **kwargs):
        """Mock retrieve method."""
        self.retrieve_calls.append({"query": query, "k": k, "kwargs": kwargs})
        
        if self.should_error:
            raise ValueError("Mock retrieval error")
        
        return self.documents[:k]


@pytest.fixture
def mock_retriever():
    """Create a mock retriever with sample documents."""
    docs = [
        Document(content=f"Document {i}", metadata={"id": i, "source": f"file{i}.txt"})
        for i in range(10)
    ]
    return MockRetriever(documents=docs)


@pytest.fixture
def mock_message():
    """Create a mock SPADE message."""
    msg = Mock(spec=Message)
    msg.body = json.dumps({"query": "test query", "k": 4})
    msg.sender = "requester@localhost"
    msg.thread = "test-thread-123"
    msg.id = "msg_123"
    return msg


class TestRetrievalBehaviourInitialization:
    """Test RetrievalBehaviour initialization."""

    def test_init_minimal(self, mock_retriever):
        """Test initialization with minimal parameters."""
        from collections import deque
        
        behaviour = RetrievalBehaviour(retriever=mock_retriever)

        assert behaviour.retriever == mock_retriever
        assert behaviour.reply_to is None
        assert behaviour.default_k == 4
        assert behaviour.on_retrieval_complete is None
        assert isinstance(behaviour._processed_messages, deque)
        assert len(behaviour._processed_messages) == 0
        assert behaviour._processed_messages.maxlen == 1000  # Check max size is set

    def test_init_full_parameters(self, mock_retriever):
        """Test initialization with all parameters."""
        callback = Mock()
        
        behaviour = RetrievalBehaviour(
            retriever=mock_retriever,
            reply_to="llm@localhost",
            default_k=10,
            on_retrieval_complete=callback
        )

        assert behaviour.retriever == mock_retriever
        assert behaviour.reply_to == "llm@localhost"
        assert behaviour.default_k == 10
        assert behaviour.on_retrieval_complete == callback

    def test_init_stats_initial_values(self, mock_retriever):
        """Test that statistics are initialized correctly."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        stats = behaviour.get_stats()

        assert stats["total_queries"] == 0
        assert stats["successful_retrievals"] == 0
        assert stats["failed_retrievals"] == 0
        assert stats["total_documents_retrieved"] == 0
        assert stats["average_retrieval_time"] == 0.0
        assert stats["last_query_time"] is None


class TestRetrievalBehaviourParseQuery:
    """Test _parse_query_message method."""

    @pytest.mark.parametrize("message_body, expected_query", [
        # Test case 1: Plain text
        ("plain text query", {"query": "plain text query"}),
        # Test case 2: Empty body
        ("", {"query": ""}),
        # Test case 3: None body
        (None, {"query": ""}),
        # Test case 4: Malformed JSON falls back to text
        ('{"invalid json}', {"query": '{"invalid json}'}),
        # Test case 5: Full JSON query
        (
            json.dumps({"query": "test", "k": 5, "filters": {"type": "doc"}}),
            {"query": "test", "k": 5, "filters": {"type": "doc"}}
        )
    ])
    def test_parse_queries(self, mock_retriever, message_body, expected_query):
        """Test parsing various query formats."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        msg = Mock(spec=Message, body=message_body)
        
        query_data = behaviour._parse_query_message(msg)

        assert query_data == expected_query


class TestRetrievalBehaviourPerformRetrieval:
    """Test _perform_retrieval method."""

    @pytest.mark.asyncio
    async def test_perform_retrieval_basic(self, mock_retriever):
        """Test basic retrieval operation."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        results = await behaviour._perform_retrieval("test query", 4, None)

        assert len(results) == 4
        assert all(isinstance(doc, Document) for doc in results)
        assert len(mock_retriever.retrieve_calls) == 1
        assert mock_retriever.retrieve_calls[0]["query"] == "test query"
        assert mock_retriever.retrieve_calls[0]["k"] == 4

    @pytest.mark.asyncio
    async def test_perform_retrieval_with_filters(self, mock_retriever):
        """Test retrieval with metadata filters."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        filters = {"source": "file1.txt"}
        await behaviour._perform_retrieval("test", 3, filters)

        assert len(mock_retriever.retrieve_calls) == 1
        call_kwargs = mock_retriever.retrieve_calls[0]["kwargs"]
        assert "filters" in call_kwargs
        assert call_kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_perform_retrieval_mmr_search(self, mock_retriever):
        """Test retrieval with MMR search type."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        await behaviour._perform_retrieval("test", 4, None, "mmr")

        call_kwargs = mock_retriever.retrieve_calls[0]["kwargs"]
        assert call_kwargs["search_type"] == "mmr"


class TestRetrievalBehaviourRun:
    """Test run method (main execution loop)."""

    @pytest.mark.asyncio
    async def test_run_no_message(self, mock_retriever):
        """Test run when no message is received."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        # Should return early without processing
        assert len(mock_retriever.retrieve_calls) == 0

    @pytest.mark.asyncio
    async def test_run_duplicate_message(self, mock_retriever, mock_message):
        """Test that duplicate messages are skipped."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        # Process message first time
        await behaviour.run()
        first_call_count = len(mock_retriever.retrieve_calls)

        # Try to process same message again
        await behaviour.run()
        second_call_count = len(mock_retriever.retrieve_calls)

        # Should not process duplicate
        assert second_call_count == first_call_count

    @pytest.mark.asyncio
    async def test_run_successful_retrieval(self, mock_retriever, mock_message):
        """Test successful retrieval flow."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have retrieved documents
        assert len(mock_retriever.retrieve_calls) == 1
        
        # Should have sent response
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_without_query_field(self, mock_retriever):
        """Test handling of message without query field."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        msg = Mock(spec=Message)
        msg.body = json.dumps({"k": 5})  # No query field
        msg.sender = "requester@localhost"
        msg.thread = "test-thread"
        msg.id = "msg_456"
        
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send error response
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "error" in sent_message.body.lower()

    @pytest.mark.asyncio
    async def test_run_with_empty_query(self, mock_retriever):
        """Test handling of message with empty query string."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        msg = Mock(spec=Message)
        msg.body = json.dumps({"query": ""})  # Empty query
        msg.sender = "requester@localhost"
        msg.thread = "test-thread"
        msg.id = "msg_empty_query"
        
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send error response
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "error" in sent_message.body.lower()
        assert "empty" in sent_message.body.lower() or "Empty" in sent_message.body

    @pytest.mark.asyncio
    async def test_run_with_whitespace_only_query(self, mock_retriever):
        """Test handling of message with whitespace-only query."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        msg = Mock(spec=Message)
        msg.body = json.dumps({"query": "   "})  # Whitespace only
        msg.sender = "requester@localhost"
        msg.thread = "test-thread"
        msg.id = "msg_whitespace_query"
        
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send error response
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "error" in sent_message.body.lower()

    @pytest.mark.asyncio
    async def test_run_with_none_query(self, mock_retriever):
        """Test handling of message with None query."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        msg = Mock(spec=Message)
        msg.body = json.dumps({"query": None})  # None query
        msg.sender = "requester@localhost"
        msg.thread = "test-thread"
        msg.id = "msg_none_query"
        
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send error response
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert "error" in sent_message.body.lower()

    @pytest.mark.asyncio
    async def test_run_with_callback(self, mock_retriever, mock_message):
        """Test that callback is called after successful retrieval."""
        callback = Mock()
        behaviour = RetrievalBehaviour(
            retriever=mock_retriever,
            on_retrieval_complete=callback
        )
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Callback should have been called
        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "test query"  # query
        assert isinstance(call_args[1], list)  # results

    @pytest.mark.asyncio
    async def test_run_updates_stats(self, mock_retriever, mock_message):
        """Test that stats are updated after retrieval."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        initial_stats = behaviour.get_stats()
        await behaviour.run()
        updated_stats = behaviour.get_stats()

        assert updated_stats["total_queries"] == initial_stats["total_queries"] + 1
        assert updated_stats["successful_retrievals"] == initial_stats["successful_retrievals"] + 1
        assert updated_stats["total_documents_retrieved"] > 0


class TestRetrievalBehaviourSendResponse:
    """Test _send_retrieval_response method."""

    @pytest.mark.asyncio
    async def test_send_response_to_original_sender(self, mock_retriever, mock_message):
        """Test sending response to original sender."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()
        
        docs = [Document(content="Test", metadata={})]
        await behaviour._send_retrieval_response(mock_message, "query", docs, 0.5)

        behaviour.send.assert_called_once()
        sent_msg = behaviour.send.call_args[0][0]
        assert sent_msg.to == str(mock_message.sender)

    @pytest.mark.asyncio
    async def test_send_response_to_reply_to(self, mock_retriever, mock_message):
        """Test sending response to reply_to address."""
        behaviour = RetrievalBehaviour(
            retriever=mock_retriever,
            reply_to="custom@localhost"
        )
        behaviour.send = AsyncMock()
        
        docs = [Document(content="Test", metadata={})]
        await behaviour._send_retrieval_response(mock_message, "query", docs, 0.5)

        behaviour.send.assert_called_once()
        sent_msg = behaviour.send.call_args[0][0]
        assert sent_msg.to == "custom@localhost"

    @pytest.mark.asyncio
    async def test_send_response_includes_documents(self, mock_retriever, mock_message):
        """Test that response includes retrieved documents."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()
        
        docs = [
            Document(content="Doc 1", metadata={"id": 1}),
            Document(content="Doc 2", metadata={"id": 2})
        ]
        await behaviour._send_retrieval_response(mock_message, "query", docs, 0.5)

        sent_msg = behaviour.send.call_args[0][0]
        response_data = json.loads(sent_msg.body)
        
        assert "documents" in response_data
        assert len(response_data["documents"]) == 2

    @pytest.mark.asyncio
    async def test_send_response_metadata(self, mock_retriever, mock_message):
        """Test that response includes correct metadata."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()

        docs = [Document(content="Test", metadata={})] * 3
        # Use a mock message object for the reply so we can inspect it
        with patch("spade_llm.behaviour.retrieval_behaviour.Message") as MockMessage:
            # When a Message is created, return a mock we can check
            mock_reply = Mock(spec=Message)
            MockMessage.return_value = mock_reply

            await behaviour._send_retrieval_response(mock_message, "test query", docs, 1.234)

            # Verify metadata was set with the correct calls
            expected_calls = [
                call("message_type", "retrieval_response"),
                call("query", "test query"),
                call("num_results", "3"),
                call("retrieval_time", "1.234"),
            ]
            mock_reply.set_metadata.assert_has_calls(expected_calls, any_order=True)
            
            # You can still check other properties
            assert mock_reply.thread == mock_message.thread
class TestRetrievalBehaviourSendError:
    """Test _send_error_response method."""

    @pytest.mark.asyncio
    async def test_send_error_response(self, mock_retriever, mock_message):
        """Test sending error response."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()

        await behaviour._send_error_response(mock_message, "Test error")

        behaviour.send.assert_called_once()
        sent_msg = behaviour.send.call_args[0][0]
        
        error_data = json.loads(sent_msg.body)
        assert "error" in error_data
        assert error_data["error"] == "Test error"

    @pytest.mark.asyncio
    async def test_send_error_includes_query(self, mock_retriever, mock_message):
        """Test that error response includes the original query."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()

        await behaviour._send_error_response(mock_message, "Error occurred")

        sent_msg = behaviour.send.call_args[0][0]
        error_data = json.loads(sent_msg.body)
        
        assert "query" in error_data
        assert error_data["query"] == mock_message.body


class TestRetrievalBehaviourUpdateStats:
    """Test _update_stats method."""

    def test_update_stats_increments_counters(self, mock_retriever):
        """Test that stats counters are incremented."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        docs = [Document(content="Test", metadata={})] * 3
        behaviour._update_stats(docs, 0.5)

        stats = behaviour.get_stats()
        assert stats["total_queries"] == 1
        assert stats["successful_retrievals"] == 1
        assert stats["total_documents_retrieved"] == 3

    def test_update_stats_average_time(self, mock_retriever):
        """Test that average retrieval time is calculated correctly."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        docs = [Document(content="Test", metadata={})]
        
        behaviour._update_stats(docs, 1.0)
        assert behaviour.get_stats()["average_retrieval_time"] == 1.0
        
        behaviour._update_stats(docs, 3.0)
        assert behaviour.get_stats()["average_retrieval_time"] == 2.0

    def test_update_stats_last_query_time(self, mock_retriever):
        """Test that last_query_time is updated."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        before_time = time.time()
        docs = [Document(content="Test", metadata={})]
        behaviour._update_stats(docs, 0.5)
        after_time = time.time()

        last_query = behaviour.get_stats()["last_query_time"]
        assert last_query is not None
        assert before_time <= last_query <= after_time


class TestRetrievalBehaviourUpdateRetriever:
    """Test update_retriever method."""

    def test_update_retriever(self, mock_retriever):
        """Test updating the retriever."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        new_retriever = MockRetriever([Document(content="New", metadata={})])
        behaviour.update_retriever(new_retriever)

        assert behaviour.retriever == new_retriever

    def test_update_retriever_logging(self, mock_retriever):
        """Test that update_retriever logs the update."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        new_retriever = MockRetriever()

        with patch("spade_llm.behaviour.retrieval_behaviour.logger") as mock_logger:
            behaviour.update_retriever(new_retriever)
            mock_logger.info.assert_called()


class TestRetrievalBehaviourSetDefaultK:
    """Test set_default_k method."""

    def test_set_default_k(self, mock_retriever):
        """Test setting default k value."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever, default_k=4)
        
        behaviour.set_default_k(10)
        assert behaviour.default_k == 10

    def test_set_default_k_logging(self, mock_retriever):
        """Test that set_default_k logs the change."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)

        with patch("spade_llm.behaviour.retrieval_behaviour.logger") as mock_logger:
            behaviour.set_default_k(8)
            mock_logger.info.assert_called()


class TestRetrievalBehaviourGetStats:
    """Test get_stats method."""

    def test_get_stats_returns_copy(self, mock_retriever):
        """Test that get_stats returns a copy of stats."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        
        stats1 = behaviour.get_stats()
        stats1["total_queries"] = 999
        
        stats2 = behaviour.get_stats()
        assert stats2["total_queries"] == 0  # Original unchanged


class TestRetrievalBehaviourErrorHandling:
    """Test error handling in RetrievalBehaviour."""

    @pytest.mark.asyncio
    async def test_run_handles_retrieval_error(self, mock_message):
        """Test that retrieval errors are handled gracefully."""
        error_retriever = MockRetriever(should_error=True)
        behaviour = RetrievalBehaviour(retriever=error_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should have sent error response
        behaviour.send.assert_called()
        sent_msg = behaviour.send.call_args[0][0]
        assert "error" in sent_msg.body.lower() or "Error" in sent_msg.body

    @pytest.mark.asyncio
    async def test_run_updates_failed_stats(self, mock_message):
        """Test that failed retrievals update stats."""
        error_retriever = MockRetriever(should_error=True)
        behaviour = RetrievalBehaviour(retriever=error_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        stats = behaviour.get_stats()
        assert stats["failed_retrievals"] > 0

    @pytest.mark.asyncio
    async def test_run_logs_retrieval_error(self, mock_message, caplog):
        """Test that retrieval errors are logged."""
        import logging
        
        error_retriever = MockRetriever(should_error=True)
        behaviour = RetrievalBehaviour(retriever=error_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        with caplog.at_level(logging.ERROR, logger="spade_llm.behaviour"):
            await behaviour.run()

        assert "Error processing retrieval request" in caplog.text
        assert "Mock retrieval error" in caplog.text  # Check exception info is logged


class TestRetrievalBehaviourEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_processed_messages_deque_limits_size(self, mock_retriever):
        """Test that processed messages deque prevents memory leak."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()
        
        # Process more than maxlen (1000) messages
        for i in range(1500):
            msg = Mock(spec=Message)
            msg.body = json.dumps({"query": f"query {i}"})
            msg.sender = "requester@localhost"
            msg.thread = f"thread-{i}"
            msg.id = f"msg_{i}"
            
            behaviour.receive = AsyncMock(return_value=msg)
            await behaviour.run()

        # Should only keep last 1000 messages
        assert len(behaviour._processed_messages) == 1000
        # Most recent message should be in deque
        assert "msg_1499" in behaviour._processed_messages
        # Oldest messages should be evicted
        assert "msg_0" not in behaviour._processed_messages

    @pytest.mark.asyncio
    async def test_run_with_zero_results(self, mock_retriever, mock_message):
        """Test handling of retrieval with zero results."""
        empty_retriever = MockRetriever(documents=[])
        behaviour = RetrievalBehaviour(retriever=empty_retriever)
        behaviour.receive = AsyncMock(return_value=mock_message)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should still send response with empty document list
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_very_large_k(self, mock_retriever):
        """Test retrieval with very large k value."""
        msg = Mock(spec=Message)
        msg.body = json.dumps({"query": "test", "k": 10000})
        msg.sender = "requester@localhost"
        msg.thread = "test-thread"
        msg.id = "msg_large_k"
        
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should process without error
        behaviour.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_unicode_query(self, mock_retriever):
        """Test handling of unicode characters in query."""
        msg = Mock(spec=Message)
        msg.body = json.dumps({"query": "Tëst quëry with ñ and 中文"})
        msg.sender = "requester@localhost"
        msg.thread = "test-thread"
        msg.id = "msg_unicode"
        
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        behaviour.send.assert_called_once()


class TestRetrievalBehaviourIntegration:
    """Integration tests for RetrievalBehaviour."""

    @pytest.mark.asyncio
    async def test_full_retrieval_flow(self, mock_retriever):
        """Test complete retrieval workflow."""
        behaviour = RetrievalBehaviour(
            retriever=mock_retriever,
            default_k=5
        )
        
        msg = Mock(spec=Message)
        msg.body = json.dumps({"query": "integration test", "k": 3})
        msg.sender = "requester@localhost"
        msg.thread = "integration-thread"
        msg.id = "msg_integration"
        
        behaviour.receive = AsyncMock(return_value=msg)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Verify full flow
        assert len(mock_retriever.retrieve_calls) == 1
        behaviour.send.assert_called_once()
        
        stats = behaviour.get_stats()
        assert stats["total_queries"] == 1
        assert stats["successful_retrievals"] == 1

    @pytest.mark.asyncio
    async def test_multiple_sequential_retrievals(self, mock_retriever):
        """Test multiple sequential retrieval operations."""
        behaviour = RetrievalBehaviour(retriever=mock_retriever)
        behaviour.send = AsyncMock()

        for i in range(3):
            msg = Mock(spec=Message)
            msg.body = json.dumps({"query": f"query {i}"})
            msg.sender = "requester@localhost"
            msg.thread = f"thread-{i}"
            msg.id = f"msg_{i}"
            
            behaviour.receive = AsyncMock(return_value=msg)
            await behaviour.run()

        stats = behaviour.get_stats()
        assert stats["total_queries"] == 3
        assert stats["successful_retrievals"] == 3
