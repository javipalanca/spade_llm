"""Base classes for vector store implementations."""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Awaitable

from ..core.document import Document

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores.
    
    Vector stores provide persistent storage and retrieval of document embeddings
    for similarity search.
    """
    def __init__(
        self,
        embedding_fn: Optional[Callable[[List[str]], Awaitable[List[List[float]]]]] = None,
    ):
        """Initialize the vector store.
        
        Args:
            embedding_fn: Optional async function that takes a list of texts and returns
                their embeddings. Example: provider.get_embeddings
        """
        self.embedding_fn = embedding_fn

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection and resources."""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Store documents with their embeddings.
        
        Args:
            documents: List of Document objects with embeddings.
            
        Returns:
            List of stored document IDs.
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """Perform a similarity search against the vector store.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of Document objects ordered by similarity score
        """
        pass

    @abstractmethod
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with explicit scores.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of (Document, score) tuples ordered by similarity score
        """
        pass

    @abstractmethod
    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs
    ) -> List[Document]:
        """Return documents selected using maximal marginal relevance.
        
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        
        Args:
            query: Search query text
            k: Number of documents to return. Defaults to 4
            fetch_k: Number of documents to fetch for MMR algorithm
            lambda_mult: Balance between diversity (0) and similarity (1). Defaults to 0.5
            **kwargs: Additional keyword arguments
            
        Returns:
            List of documents selected by maximal marginal relevance
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_document_count(self) -> int:
        """Get the total number of documents in the store.
        
        Returns:
            Number of documents in the store
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connections.
        """
        pass