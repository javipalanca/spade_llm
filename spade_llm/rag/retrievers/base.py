"""Base classes for retriever implementations."""

import logging
from abc import ABC, abstractmethod
from typing import List

from ..core.document import Document

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for document retrievers.
    
    Retrievers are responsible for finding and returning relevant documents
    based on a query. Different retriever implementations can use various
    strategies such as vector similarity, keyword matching etc.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents relevant to the query.
        
        Args:
            query: The search query
            k: Maximum number of documents to retrieve
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            List of Document objects ordered by relevance
        """
        pass
