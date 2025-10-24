"""Vector store retriever implementation."""

import logging
from typing import List, Tuple, Union, Literal

from .base import BaseRetriever
from ..core.document import Document
from ..vector_stores.base import VectorStore

logger = logging.getLogger(__name__)

# Type alias for search types
SearchType = Literal["similarity", "similarity_score", "mmr"]


class VectorStoreRetriever(BaseRetriever):
    """Retriever implementation using vector stores for similarity search.
    
    This retriever wraps a vector store and provides document retrieval
    based on vector similarity search with support for different search types:
    - similarity: Standard similarity search returning documents
    - similarity_score: Similarity search with relevance scores
    - mmr: Maximal Marginal Relevance for diversity
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
        search_type: SearchType = "similarity",
        **kwargs
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Retrieve documents using the specified search type.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            search_type: Type of search to perform:
                - "similarity": Standard similarity search (default)
                - "similarity_score": Similarity search with scores
                - "mmr": Maximal Marginal Relevance search
            **kwargs: Additional search parameters:
                - filters: Metadata filters (dict)
                - fetch_k: Number of docs to fetch for MMR (int, default 20)
                - lambda_mult: MMR diversity parameter (float, default 0.5)
                - Other vector store specific parameters
            
        Returns:
            - List[Document] for "similarity" and "mmr" search types
            - List[Tuple[Document, float]] for "similarity_score" search type
        """
        try:
            logger.debug(
                f"Retrieving {k} documents for query: {query[:100]}... "
                f"(search_type={search_type})"
            )
            
            if search_type == "similarity":
                # Standard similarity search
                documents = await self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    **kwargs
                )
                logger.debug(f"Retrieved {len(documents)} documents")
                return documents
                
            elif search_type == "similarity_score":
                # Similarity search with scores
                documents_with_scores = await self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    **kwargs
                )
                logger.debug(f"Retrieved {len(documents_with_scores)} documents with scores")
                return documents_with_scores
                
            elif search_type == "mmr":
                # Maximal Marginal Relevance search
                if not hasattr(self.vector_store, "max_marginal_relevance_search"):
                    logger.warning(
                        f"Vector store {type(self.vector_store).__name__} does not support MMR. "
                        f"Falling back to similarity search."
                    )
                    documents = await self.vector_store.similarity_search(
                        query=query,
                        k=k,
                        **kwargs
                    )
                    return documents
                
                # Extract MMR-specific parameters
                fetch_k = kwargs.pop("fetch_k", 20)
                lambda_mult = kwargs.pop("lambda_mult", 0.5)
                
                documents = await self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    **kwargs
                )
                logger.debug(f"Retrieved {len(documents)} documents using MMR")
                return documents
                
            else:
                raise ValueError(
                    f"Invalid search_type: {search_type}. "
                    f"Must be one of: 'similarity', 'similarity_score', 'mmr'"
                )
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise