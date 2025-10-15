"""Tests for base text splitter."""

import pytest
from spade_llm.rag import TextSplitter, Document


class TestTextSplitter:
    """Test cases for TextSplitter base class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        # Simple implementation for testing
        class SimpleTextSplitter(TextSplitter):
            def split_text(self, text: str):
                return [text]
            
            def split_documents(self, documents):
                return [Document(content=chunk, metadata=doc.metadata.copy()) 
                        for doc in documents for chunk in self.split_text(doc.content)]
        
        splitter = SimpleTextSplitter()
        assert splitter.chunk_size == 2000
        assert splitter.chunk_overlap == 200

    def test_initialization_validation(self):
        """Test parameter validation."""
        class SimpleTextSplitter(TextSplitter):
            def split_text(self, text: str):
                return [text]
            
            def split_documents(self, documents):
                return [Document(content=chunk, metadata=doc.metadata.copy()) 
                        for doc in documents for chunk in self.split_text(doc.content)]
        
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            SimpleTextSplitter(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            SimpleTextSplitter(chunk_overlap=-1)

        with pytest.raises(ValueError, match="larger chunk overlap.*should be smaller"):
            SimpleTextSplitter(chunk_size=100, chunk_overlap=150)
