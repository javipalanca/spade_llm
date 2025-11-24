"""Document retrieval implementations.

This module provides retriever implementations for finding and returning
relevant documents based on queries. Different retrievers use different strategies
like vector similarity, keyword matching, etc.
"""

from .base import BaseRetriever
from .vector_store import VectorStoreRetriever

__all__ = ["BaseRetriever", "VectorStoreRetriever"]