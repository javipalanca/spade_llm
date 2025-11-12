"""Tests for RetrievalAgent class."""

import pytest
from unittest.mock import Mock, patch

from spade_llm.agent.retrieval_agent import RetrievalAgent
from spade_llm.behaviour.retrieval_behaviour import RetrievalBehaviour
from spade_llm.rag.retrievers.base import BaseRetriever
from spade_llm.rag.core.document import Document


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    def __init__(self, documents=None):
        self.documents = documents or []
        self.retrieve_calls = []

    async def retrieve(self, query: str, k: int = 4, **kwargs):
        """Mock retrieve method."""
        self.retrieve_calls.append({"query": query, "k": k, "kwargs": kwargs})
        return self.documents[:k]


@pytest.fixture
def mock_retriever():
    """Create a mock retriever with sample documents."""
    docs = [
        Document(content=f"Document {i}", metadata={"id": i})
        for i in range(10)
    ]
    return MockRetriever(documents=docs)


@pytest.fixture
def retrieval_callback():
    """Create a mock callback for retrieval completion."""
    callback = Mock()
    return callback


class TestRetrievalAgentInitialization:
    """Test RetrievalAgent initialization."""

    def test_init_minimal(self, mock_retriever):
        """Test initialization with minimal parameters."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        assert agent.jid == "retrieval@localhost"
        assert agent.password == "password"
        assert agent.retriever == mock_retriever
        assert agent.reply_to is None
        assert agent.default_k == 4
        assert agent.on_retrieval_complete is None
        assert isinstance(agent.retrieval_behaviour, RetrievalBehaviour)

    def test_init_full_parameters(self, mock_retriever, retrieval_callback):
        """Test initialization with all parameters."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever,
            reply_to="llm@localhost",
            default_k=10,
            on_retrieval_complete=retrieval_callback,
            verify_security=False
        )

        assert agent.jid == "retrieval@localhost"
        assert agent.retriever == mock_retriever
        assert agent.reply_to == "llm@localhost"
        assert agent.default_k == 10
        assert agent.on_retrieval_complete == retrieval_callback
        assert isinstance(agent.retrieval_behaviour, RetrievalBehaviour)

    def test_behaviour_initialization(self, mock_retriever):
        """Test that behaviour is properly initialized with agent parameters."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever,
            default_k=8
        )

        behaviour = agent.retrieval_behaviour
        assert behaviour.retriever == mock_retriever
        assert behaviour.default_k == 8

    def test_init_with_custom_callback(self, mock_retriever):
        """Test initialization with custom callback."""
        def custom_callback(query, results):
            pass  # Callback with correct signature

        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever,
            on_retrieval_complete=custom_callback
        )

        assert agent.on_retrieval_complete == custom_callback
        assert agent.retrieval_behaviour.on_retrieval_complete == custom_callback


class TestRetrievalAgentSetup:
    """Test RetrievalAgent setup method."""

    @pytest.mark.asyncio
    async def test_setup_adds_behaviour(self, mock_retriever):
        """Test that setup adds the retrieval behaviour."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        # Mock the add_behaviour method
        agent.add_behaviour = Mock()

        await agent.setup()

        # Verify behaviour was added
        agent.add_behaviour.assert_called_once()
        call_args = agent.add_behaviour.call_args
        
        # First argument should be the behaviour
        assert call_args[0][0] == agent.retrieval_behaviour
        
        # Second argument should be a template with message_type metadata
        template = call_args[0][1]
        assert template is not None

    @pytest.mark.asyncio
    async def test_setup_logging(self, mock_retriever):
        """Test that setup logs agent startup."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )
        agent.add_behaviour = Mock()

        with patch("spade_llm.agent.retrieval_agent.logger") as mock_logger:
            await agent.setup()
            mock_logger.info.assert_called()


class TestRetrievalAgentUpdateRetriever:
    """Test update_retriever method."""

    def test_update_retriever(self, mock_retriever):
        """Test updating the retriever."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        # Create new retriever
        new_docs = [Document(content="New doc", metadata={"new": True})]
        new_retriever = MockRetriever(documents=new_docs)

        # Update retriever
        agent.update_retriever(new_retriever)

        assert agent.retriever == new_retriever
        assert agent.retrieval_behaviour.retriever == new_retriever

    def test_update_retriever_logging(self, mock_retriever):
        """Test that update_retriever logs the update."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )
        new_retriever = MockRetriever()

        with patch("spade_llm.agent.retrieval_agent.logger") as mock_logger:
            agent.update_retriever(new_retriever)
            mock_logger.info.assert_called()


class TestRetrievalAgentSetDefaultK:
    """Test set_default_k method."""

    def test_set_default_k(self, mock_retriever):
        """Test setting default k value."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever,
            default_k=4
        )

        assert agent.default_k == 4 and agent.retrieval_behaviour.default_k == 4

        agent.set_default_k(10)

        assert agent.default_k == 10 and agent.retrieval_behaviour.default_k == 10

    def test_set_default_k_logging(self, mock_retriever):
        """Test that set_default_k logs the change."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        with patch("spade_llm.agent.retrieval_agent.logger") as mock_logger:
            agent.set_default_k(8)
            mock_logger.info.assert_called()


class TestRetrievalAgentGetStats:
    """Test get_retrieval_stats method."""

    def test_get_stats_returns_dict(self, mock_retriever):
        """Test that get_retrieval_stats returns a dictionary."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        stats = agent.get_retrieval_stats()

        assert isinstance(stats, dict)

    def test_get_stats_contains_expected_keys(self, mock_retriever):
        """Test that stats contain expected keys."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        stats = agent.get_retrieval_stats()

        expected_keys = [
            "total_queries",
            "successful_retrievals",
            "failed_retrievals",
            "total_documents_retrieved",
            "average_retrieval_time"
        ]
        
        for key in expected_keys:
            assert key in stats

    def test_get_stats_initial_values(self, mock_retriever):
        """Test that stats have correct initial values."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        stats = agent.get_retrieval_stats()

        assert stats["total_queries"] == 0
        assert stats["successful_retrievals"] == 0
        assert stats["failed_retrievals"] == 0
        assert stats["total_documents_retrieved"] == 0
        assert stats["average_retrieval_time"] == 0.0


class TestRetrievalAgentIntegration:
    """Integration tests for RetrievalAgent."""

    def test_agent_inherits_from_agent(self, mock_retriever):
        """Test that RetrievalAgent inherits from SPADE Agent."""
        from spade.agent import Agent

        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        assert isinstance(agent, Agent)

    def test_behaviour_configuration_propagation(self, mock_retriever, retrieval_callback):
        """Test that configuration propagates to behaviour."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever,
            reply_to="llm@localhost",
            default_k=7,
            on_retrieval_complete=retrieval_callback
        )

        behaviour = agent.retrieval_behaviour
        
        assert behaviour.retriever == mock_retriever
        assert behaviour.reply_to == "llm@localhost"
        assert behaviour.default_k == 7
        assert behaviour.on_retrieval_complete == retrieval_callback

    def test_multiple_retriever_updates(self, mock_retriever):
        """Test multiple retriever updates."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        retriever1 = MockRetriever([Document(content="Set 1", metadata={})])
        retriever2 = MockRetriever([Document(content="Set 2", metadata={})])
        retriever3 = MockRetriever([Document(content="Set 3", metadata={})])

        agent.update_retriever(retriever1)
        assert agent.retriever == retriever1

        agent.update_retriever(retriever2)
        assert agent.retriever == retriever2

        agent.update_retriever(retriever3)
        assert agent.retriever == retriever3


class TestRetrievalAgentEdgeCases:
    """Test edge cases and error handling."""

    def test_set_default_k_to_zero(self, mock_retriever):
        """Test setting default k to zero."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        agent.set_default_k(0)
        assert agent.default_k == 0

    def test_set_default_k_to_large_number(self, mock_retriever):
        """Test setting default k to a very large number."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever
        )

        agent.set_default_k(10000)
        assert agent.default_k == 10000

    def test_none_callback(self, mock_retriever):
        """Test that None callback doesn't cause issues."""
        agent = RetrievalAgent(
            jid="retrieval@localhost",
            password="password",
            retriever=mock_retriever,
            on_retrieval_complete=None
        )

        assert agent.on_retrieval_complete is None
        assert agent.retrieval_behaviour.on_retrieval_complete is None