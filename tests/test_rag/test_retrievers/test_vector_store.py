"""Tests for vector store retriever."""

import pytest
from unittest.mock import Mock, AsyncMock

from spade_llm.rag import (
    Document,
    VectorStoreRetriever,
)


class TestVectorStoreRetriever:
    """Test cases for VectorStoreRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock = Mock()
        mock.similarity_search = AsyncMock()
        mock.similarity_search_with_score = AsyncMock()
        return mock

    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create a VectorStoreRetriever instance."""
        return VectorStoreRetriever(vector_store=mock_vector_store)

    def test_initialization(self, mock_vector_store):
        """Test retriever initialization."""
        retriever = VectorStoreRetriever(vector_store=mock_vector_store)
        assert retriever.vector_store == mock_vector_store

    @pytest.mark.asyncio
    async def test_retrieve_success(self, retriever, mock_vector_store):
        """Test successful document retrieval."""
        expected_docs = [
            Document(content="Result 1", metadata={"id": 1}),
            Document(content="Result 2", metadata={"id": 2}),
        ]
        mock_vector_store.similarity_search.return_value = expected_docs

        results = await retriever.retrieve("test query", k=2)

        assert results == expected_docs
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=2
        )

    @pytest.mark.asyncio
    async def test_retrieve_with_kwargs(self, retriever, mock_vector_store):
        """Test retrieval with additional kwargs."""
        mock_vector_store.similarity_search.return_value = []

        await retriever.retrieve(
            "test query",
            k=5,
            filters={"category": "science"}
        )

        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=5,
            filters={"category": "science"}
        )

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, retriever, mock_vector_store):
        """Test retrieval with no results."""
        mock_vector_store.similarity_search.return_value = []

        results = await retriever.retrieve("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_error_handling(self, retriever, mock_vector_store):
        """Test error handling during retrieval."""
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")

        with pytest.raises(Exception, match="Search failed"):
            await retriever.retrieve("test query")

    @pytest.mark.asyncio
    async def test_retrieve_with_scores(self, retriever, mock_vector_store):
        """Test retrieval with similarity scores."""
        expected_results = [
            (Document(content="Result 1", metadata={"id": 1}), 0.95),
            (Document(content="Result 2", metadata={"id": 2}), 0.85),
        ]
        mock_vector_store.similarity_search_with_score.return_value = expected_results

        results = await retriever.retrieve_with_scores("test query", k=2)

        assert results == expected_results
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            query="test query",
            k=2
        )

    @pytest.mark.asyncio
    async def test_retrieve_default_k_value(self, retriever, mock_vector_store):
        """Test retrieval uses default k value."""
        mock_vector_store.similarity_search.return_value = []

        await retriever.retrieve("test query")

        # Default k should be 4
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=4
        )
