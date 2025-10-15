"""Tests for base document loader."""

import pytest

from spade_llm.rag.document_loaders.base import BaseDocumentLoader
from spade_llm.rag.core.document import Document


class TestBaseDocumentLoader:
    """Test cases for the abstract BaseDocumentLoader."""

    def test_base_class_extension_support(self):
        """Test that the base class itself supports no extensions."""
        assert BaseDocumentLoader.get_supported_extensions() == set()
        assert not BaseDocumentLoader.supports_extension('.txt')

    @pytest.mark.asyncio
    async def test_load_is_consumer_of_load_stream(self):
        """Test that the base load() correctly consumes load_stream()."""
        class DummyLoader(BaseDocumentLoader):
            async def load_stream(self):
                yield Document(content="doc1", metadata={})
                yield Document(content="doc2", metadata={})
        
        loader = DummyLoader()
        documents = await loader.load()
        
        assert len(documents) == 2
        assert documents[0].content == "doc1"
