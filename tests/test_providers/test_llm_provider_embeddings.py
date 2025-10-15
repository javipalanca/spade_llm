"""Tests for LLMProvider embeddings functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from spade_llm.providers.llm_provider import LLMProvider, ModelFormat
from openai import OpenAIError


class TestGetEmbeddings:
    """Test the get_embeddings method."""

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_single_text(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings for a single text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock embedding response
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.1] * 384
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        embeddings = await provider.get_embeddings(["test text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert all(isinstance(val, float) for val in embeddings[0])

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_multiple_texts(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings for multiple texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock multiple embedding responses
        mock_embedding_items = [
            Mock(embedding=[0.1] * 384),
            Mock(embedding=[0.2] * 384),
            Mock(embedding=[0.3] * 384),
        ]
        
        mock_response = Mock()
        mock_response.data = mock_embedding_items
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        texts = ["text 1", "text 2", "text 3"]
        embeddings = await provider.get_embeddings(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Verify the API was called with correct parameters
        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args
        assert call_args[1]['model'] == 'gpt-4o-mini'
        assert call_args[1]['input'] == texts

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_empty_list(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings for empty list."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = []
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        embeddings = await provider.get_embeddings([])
        
        assert embeddings == []

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_with_ollama(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings with Ollama provider."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.5] * 768  # Different dimension
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider.create_ollama(model="nomic-embed-text")
        embeddings = await provider.get_embeddings(["test text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        
        # Verify model name was prepared correctly (strip ollama/ prefix)
        call_args = mock_to_thread.call_args
        assert call_args[1]['model'] == 'nomic-embed-text'

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_openai_error(self, mock_openai_class, mock_to_thread):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_to_thread.side_effect = OpenAIError("API Error")
        
        provider = LLMProvider()
        
        with pytest.raises(OpenAIError):
            await provider.get_embeddings(["test"])

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_unexpected_error(self, mock_openai_class, mock_to_thread):
        """Test handling of unexpected errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_to_thread.side_effect = ValueError("Unexpected error")
        
        provider = LLMProvider()
        
        with pytest.raises(ValueError):
            await provider.get_embeddings(["test"])

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_with_long_text(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings for long text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.1] * 1536  # OpenAI dimension
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        long_text = "word " * 1000  # Very long text
        embeddings = await provider.get_embeddings([long_text])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_with_special_characters(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings for text with special characters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.2] * 384
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        special_text = "Text with émojis 🎉 and spëcial çhars!"
        embeddings = await provider.get_embeddings([special_text])
        
        assert len(embeddings) == 1
        assert all(isinstance(val, float) for val in embeddings[0])

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_batch_processing(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings for a large batch of texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock 100 embedding responses
        num_texts = 100
        mock_embedding_items = [
            Mock(embedding=[float(i % 10) / 10] * 384)
            for i in range(num_texts)
        ]
        
        mock_response = Mock()
        mock_response.data = mock_embedding_items
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        texts = [f"Document {i}" for i in range(num_texts)]
        embeddings = await provider.get_embeddings(texts)
        
        assert len(embeddings) == num_texts
        assert all(len(emb) == 384 for emb in embeddings)

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_different_dimensions(self, mock_openai_class, mock_to_thread):
        """Test that embeddings maintain their dimension."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Test with different embedding dimensions
        for dim in [384, 768, 1536]:
            mock_embedding_item = Mock()
            mock_embedding_item.embedding = [0.5] * dim
            
            mock_response = Mock()
            mock_response.data = [mock_embedding_item]
            
            mock_to_thread.return_value = mock_response
            
            provider = LLMProvider()
            embeddings = await provider.get_embeddings(["test"])
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == dim

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_get_embeddings_with_custom_model(self, mock_openai_class, mock_to_thread):
        """Test getting embeddings with a custom model."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.3] * 384
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        # Use a custom embedding model
        provider = LLMProvider(model="text-embedding-ada-002")
        embeddings = await provider.get_embeddings(["test"])
        
        assert len(embeddings) == 1
        
        # Verify correct model was used
        call_args = mock_to_thread.call_args
        assert call_args[1]['model'] == 'text-embedding-ada-002'


class TestEmbeddingsIntegration:
    """Integration tests for embeddings with vector stores."""

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_embeddings_as_callback(self, mock_openai_class, mock_to_thread):
        """Test using get_embeddings as a callback for vector stores."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock embedding response
        mock_embedding_items = [
            Mock(embedding=[0.1] * 384),
            Mock(embedding=[0.2] * 384),
        ]
        
        mock_response = Mock()
        mock_response.data = mock_embedding_items
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        
        # Simulate vector store usage
        texts = ["doc 1", "doc 2"]
        embeddings = await provider.get_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_embeddings_consistency(self, mock_openai_class, mock_to_thread):
        """Test that embeddings are consistent across calls."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Create deterministic mock embeddings
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.5] * 384
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider()
        
        # Call twice with same text
        embeddings1 = await provider.get_embeddings(["test"])
        embeddings2 = await provider.get_embeddings(["test"])
        
        # Both should have same dimension (mocked to be same)
        assert len(embeddings1[0]) == len(embeddings2[0])


class TestEmbeddingsProviderSpecific:
    """Test embeddings with different provider configurations."""

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_openai_embeddings(self, mock_openai_class, mock_to_thread):
        """Test embeddings with OpenAI provider."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.1] * 1536
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider.create_openai(
            api_key="test-key",
            model="text-embedding-ada-002"
        )
        
        embeddings = await provider.get_embeddings(["OpenAI test"])
        
        assert len(embeddings) == 1
        assert provider.provider_name == "OpenAI"

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_ollama_embeddings_model_preparation(self, mock_openai_class, mock_to_thread):
        """Test that Ollama model names are prepared correctly for embeddings."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.2] * 768
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider.create_ollama(
            model="mxbai-embed-large",
            base_url="http://localhost:11434/v1"
        )
        
        embeddings = await provider.get_embeddings(["Ollama test"])
        
        assert len(embeddings) == 1
        
        # Verify the model name was prepared (ollama/ prefix stripped)
        call_args = mock_to_thread.call_args
        assert call_args[1]['model'] == 'mxbai-embed-large'

    @patch('spade_llm.providers.llm_provider.asyncio.to_thread')
    @patch('spade_llm.providers.llm_provider.OpenAI')
    @pytest.mark.asyncio
    async def test_vllm_embeddings(self, mock_openai_class, mock_to_thread):
        """Test embeddings with vLLM provider."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_item = Mock()
        mock_embedding_item.embedding = [0.3] * 384
        
        mock_response = Mock()
        mock_response.data = [mock_embedding_item]
        
        mock_to_thread.return_value = mock_response
        
        provider = LLMProvider.create_vllm(
            model="embedding-model",
            base_url="http://localhost:8000/v1"
        )
        
        embeddings = await provider.get_embeddings(["vLLM test"])
        
        assert len(embeddings) == 1
        assert provider.provider_name == "vLLM"
