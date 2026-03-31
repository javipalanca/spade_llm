"""Tests for LLMProvider embeddings functionality using LiteLLM."""

from unittest.mock import Mock, patch

import pytest

from spade_llm.providers.llm_provider import LLMProvider


class TestGetEmbeddings:
    """Test the get_embeddings method."""

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_single_text(self, mock_aembedding):
        """Test getting embeddings for a single text."""
        mock_embedding_item = {"embedding": [0.1] * 384}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        embeddings = await provider.get_embeddings(["test text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert all(isinstance(val, float) for val in embeddings[0])

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_multiple_texts(self, mock_aembedding):
        """Test getting embeddings for multiple texts."""
        mock_embedding_items = [
            {"embedding": [0.1] * 384},
            {"embedding": [0.2] * 384},
            {"embedding": [0.3] * 384},
        ]
        mock_response = Mock()
        mock_response.data = mock_embedding_items
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        texts = ["text 1", "text 2", "text 3"]
        embeddings = await provider.get_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

        mock_aembedding.assert_called_once()
        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["model"] == "text-embedding-ada-002"
        assert call_kwargs["input"] == texts

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_empty_list(self, mock_aembedding):
        """Test getting embeddings for empty list."""
        mock_response = Mock()
        mock_response.data = []
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        embeddings = await provider.get_embeddings([])

        assert embeddings == []

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_with_ollama(self, mock_aembedding):
        """Test getting embeddings with Ollama provider."""
        mock_embedding_item = {"embedding": [0.5] * 768}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(
            model="ollama/nomic-embed-text",
            base_url="http://localhost:11434",
        )
        embeddings = await provider.get_embeddings(["test text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["model"] == "ollama/nomic-embed-text"
        assert call_kwargs["api_base"] == "http://localhost:11434"

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_api_error(self, mock_aembedding):
        """Test handling of API errors."""
        mock_aembedding.side_effect = Exception("API Error")

        provider = LLMProvider(model="text-embedding-ada-002")

        with pytest.raises(Exception, match="API Error"):
            await provider.get_embeddings(["test"])

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_unexpected_error(self, mock_aembedding):
        """Test handling of unexpected errors."""
        mock_aembedding.side_effect = ValueError("Unexpected error")

        provider = LLMProvider(model="text-embedding-ada-002")

        with pytest.raises(ValueError):
            await provider.get_embeddings(["test"])

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_with_long_text(self, mock_aembedding):
        """Test getting embeddings for long text."""
        mock_embedding_item = {"embedding": [0.1] * 1536}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        long_text = "word " * 1000
        embeddings = await provider.get_embeddings([long_text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_with_special_characters(self, mock_aembedding):
        """Test getting embeddings for text with special characters."""
        mock_embedding_item = {"embedding": [0.2] * 384}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        special_text = "Text with émojis 🎉 and spëcial çhars!"
        embeddings = await provider.get_embeddings([special_text])

        assert len(embeddings) == 1
        assert all(isinstance(val, float) for val in embeddings[0])

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_batch_processing(self, mock_aembedding):
        """Test getting embeddings for a large batch of texts."""
        num_texts = 100
        mock_embedding_items = [{"embedding": [float(i % 10) / 10] * 384} for i in range(num_texts)]
        mock_response = Mock()
        mock_response.data = mock_embedding_items
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        texts = [f"Document {i}" for i in range(num_texts)]
        embeddings = await provider.get_embeddings(texts)

        assert len(embeddings) == num_texts
        assert all(len(emb) == 384 for emb in embeddings)

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_different_dimensions(self, mock_aembedding):
        """Test that embeddings maintain their dimension."""
        for dim in [384, 768, 1536]:
            mock_embedding_item = {"embedding": [0.5] * dim}
            mock_response = Mock()
            mock_response.data = [mock_embedding_item]
            mock_aembedding.return_value = mock_response

            provider = LLMProvider(model="text-embedding-ada-002")
            embeddings = await provider.get_embeddings(["test"])

            assert len(embeddings) == 1
            assert len(embeddings[0]) == dim

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_with_custom_model(self, mock_aembedding):
        """Test getting embeddings with a custom model."""
        mock_embedding_item = {"embedding": [0.3] * 384}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")
        embeddings = await provider.get_embeddings(["test"])

        assert len(embeddings) == 1

        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["model"] == "text-embedding-ada-002"

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_passes_api_key(self, mock_aembedding):
        """Test that api_key is passed to litellm."""
        mock_embedding_item = {"embedding": [0.1] * 384}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002", api_key="test-key")
        await provider.get_embeddings(["test"])

        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["api_key"] == "test-key"

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_get_embeddings_passes_base_url(self, mock_aembedding):
        """Test that base_url is passed as api_base to litellm."""
        mock_embedding_item = {"embedding": [0.1] * 768}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(
            model="ollama/nomic-embed-text",
            base_url="http://localhost:11434",
        )
        await provider.get_embeddings(["test"])

        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"


class TestEmbeddingsIntegration:
    """Integration tests for embeddings."""

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_embeddings_as_callback(self, mock_aembedding):
        """Test using get_embeddings as a callback for vector stores."""
        mock_embedding_items = [
            {"embedding": [0.1] * 384},
            {"embedding": [0.2] * 384},
        ]
        mock_response = Mock()
        mock_response.data = mock_embedding_items
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")

        texts = ["doc 1", "doc 2"]
        embeddings = await provider.get_embeddings(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)

    @patch("spade_llm.providers.llm_provider.litellm.aembedding")
    @pytest.mark.asyncio
    async def test_embeddings_consistency(self, mock_aembedding):
        """Test that embeddings are consistent across calls."""
        mock_embedding_item = {"embedding": [0.5] * 384}
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        mock_aembedding.return_value = mock_response

        provider = LLMProvider(model="text-embedding-ada-002")

        embeddings1 = await provider.get_embeddings(["test"])
        embeddings2 = await provider.get_embeddings(["test"])

        assert len(embeddings1[0]) == len(embeddings2[0])
