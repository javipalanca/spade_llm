"""Tests for base vector store."""

import pytest
from spade_llm.rag import VectorStore


class TestVectorStore:
    """Tests for VectorStore abstract base class."""
    
    def test_is_abstract(self):
        """Test that VectorStore is abstract."""
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore
