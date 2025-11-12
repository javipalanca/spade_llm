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

        results = await retriever.retrieve("test query", k=2, search_type="similarity")

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
            search_type="similarity",
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

        results = await retriever.retrieve("test query", search_type="similarity")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_error_handling(self, retriever, mock_vector_store):
        """Test error handling during retrieval."""
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")

        with pytest.raises(Exception, match="Search failed"):
            await retriever.retrieve("test query", search_type="similarity")

    @pytest.mark.asyncio
    async def test_retrieve_similarity_with_threshold(self, retriever, mock_vector_store):
        """Test retrieve_similarity with score threshold filtering."""
        docs_with_scores = [
            (Document(content="Result 1", metadata={"id": 1}), 0.95),
            (Document(content="Result 2", metadata={"id": 2}), 0.75),
            (Document(content="Result 3", metadata={"id": 3}), 0.65),
        ]
        mock_vector_store.similarity_search_with_score.return_value = docs_with_scores

        # Only documents with score >= 0.8 should be returned
        results = await retriever.retrieve_similarity("test query", k=3, sim_threshold=0.8)

        assert len(results) == 1
        assert results[0].content == "Result 1"
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            query="test query",
            k=3
        )

    @pytest.mark.asyncio
    async def test_retrieve_default_k_value(self, retriever, mock_vector_store):
        """Test retrieval uses default k value."""
        mock_vector_store.similarity_search.return_value = []

        await retriever.retrieve("test query", search_type="similarity")

        # Default k should be 4
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=4
        )

    @pytest.mark.asyncio
    async def test_retrieve_default_search_type(self, retriever, mock_vector_store):
        """Test retrieval uses default search_type='similarity'."""
        mock_vector_store.similarity_search.return_value = []

        await retriever.retrieve("test query")

        # Default search_type should be "similarity"
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=4
        )

    @pytest.mark.asyncio
    async def test_retrieve_mmr_search(self, retriever, mock_vector_store):
        """Test MMR search type."""
        expected_docs = [
            Document(content="Result 1", metadata={"id": 1}),
            Document(content="Result 2", metadata={"id": 2}),
        ]
        mock_vector_store.max_marginal_relevance_search = AsyncMock(return_value=expected_docs)

        results = await retriever.retrieve(
            "test query",
            k=2,
            search_type="mmr",
            fetch_k=10,
            lambda_mult=0.7
        )

        assert results == expected_docs
        mock_vector_store.max_marginal_relevance_search.assert_called_once_with(
            query="test query",
            k=2,
            fetch_k=10,
            lambda_mult=0.7
        )

    @pytest.mark.asyncio
    async def test_retrieve_mmr_not_supported_raises_error(self, mock_vector_store):
        """Test MMR raises NotImplementedError when not supported by vector store."""
        # Create a mock that doesn't have the max_marginal_relevance_search attribute
        class LimitedVectorStore:
            async def similarity_search(self, query, k, **kwargs):
                return [Document(content="Result 1", metadata={"id": 1})]
        
        limited_store = LimitedVectorStore()
        retriever = VectorStoreRetriever(vector_store=limited_store)  # type: ignore
        
        with pytest.raises(NotImplementedError, match="MMR search is not supported"):
            await retriever.retrieve("test query", k=2, search_type="mmr")

    @pytest.mark.asyncio
    async def test_retrieve_invalid_search_type(self, retriever, mock_vector_store):
        """Test error handling for invalid search type."""
        with pytest.raises(ValueError, match="Invalid search_type"):
            await retriever.retrieve("test query", search_type="invalid_type")

    @pytest.mark.asyncio
    async def test_retrieve_similarity_with_filters(self, retriever, mock_vector_store):
        """Test retrieve_similarity with metadata filters."""
        expected_results = [
            Document(content="Result 1", metadata={"category": "science"}),
        ]
        mock_vector_store.similarity_search.return_value = expected_results

        results = await retriever.retrieve_similarity(
            "test query",
            k=1,
            filters={"category": "science"}
        )

        assert results == expected_results
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=1,
            filters={"category": "science"}
        )

    @pytest.mark.asyncio
    async def test_retrieve_mmr_with_custom_params(self, retriever, mock_vector_store):
        """Test retrieve_mmr with custom fetch_k and lambda_mult parameters."""
        expected_docs = [
            Document(content="Result 1", metadata={"id": 1}),
        ]
        mock_vector_store.max_marginal_relevance_search = AsyncMock(return_value=expected_docs)

        results = await retriever.retrieve_mmr(
            "test query",
            k=1,
            fetch_k=15,
            lambda_mult=0.8,
            filters={"type": "article"}
        )

        assert results == expected_docs
        mock_vector_store.max_marginal_relevance_search.assert_called_once_with(
            query="test query",
            k=1,
            fetch_k=15,
            lambda_mult=0.8,
            filters={"type": "article"}
        )

    @pytest.mark.asyncio
    async def test_retrieve_similarity_without_filters(self, retriever, mock_vector_store):
        """Test retrieve_similarity without filters uses kwargs correctly."""
        expected_results = [Document(content="Result 1", metadata={})]
        mock_vector_store.similarity_search.return_value = expected_results

        results = await retriever.retrieve_similarity("test query", k=2)

        assert results == expected_results
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=2
        )

