"""Tests for base retriever."""

import pytest
from abc import ABC

from spade_llm.rag.retrievers.base import BaseRetriever


class TestBaseRetriever:
    """Test cases for the abstract BaseRetriever base class."""

    def test_is_abstract_class(self):
        """Test that BaseRetriever is an abstract class."""
        assert issubclass(BaseRetriever, ABC)
        
        # Attempting to instantiate the abstract class should raise TypeError
        with pytest.raises(TypeError):
            BaseRetriever()

    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseRetriever.__abstractmethods__
        expected_methods = {'retrieve'}
        
        assert abstract_methods == expected_methods
