"""Vector store retriever implementation."""

import logging
from typing import List, Tuple

from .base import BaseRetriever
from ..core.document import Document
from ..vector_stores.base import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """Retriever implementation using vector stores for similarity search.
    
    This retriever wraps a vector store and provides document retrieval
    based on vector similarity search.
    """
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the vector store retriever.
        
        Args:
            vector_store: The vector store to retrieve from
        """
        self.vector_store = vector_store
        logger.info(f"Initialized VectorStoreRetriever with {type(vector_store).__name__}")
    
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents similar to the query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            **kwargs: Additional search parameters passed to vector store
            
        Returns:
            List of Document objects ordered by relevance
        """
        try:
            logger.debug(f"Retrieving {k} documents for query: {query[:100]}...")
            
            # Use the vector store's similarity search
            documents = await self.vector_store.similarity_search(
                query=query,
                k=k,
                **kwargs
            )
            
            logger.debug(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            **kwargs: Additional search parameters passed to vector store
            
        Returns:
            List of (Document, score) tuples ordered by relevance
        """
        try:
            logger.debug(f"Retrieving {k} documents with scores for query: {query[:100]}...")
            
            # Use the vector store's similarity search with scores
            documents_with_scores = await self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                **kwargs
            )
            
            logger.debug(f"Retrieved {len(documents_with_scores)} documents with scores")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error during retrieval with scores: {e}")
            raise