"""Vector store retriever implementation."""

import logging
from typing import Optional, Dict, List,Literal, Any

from .base import BaseRetriever
from ..core.document import Document
from ..vector_stores.base import VectorStore

logger = logging.getLogger(__name__)

# Type alias for search types
SearchType = Literal["similarity", "mmr"]

class VectorStoreRetriever(BaseRetriever):
    """Retriever implementation using vector stores for similarity search.
    
    This retriever wraps a vector store and provides document retrieval
    based on vector similarity search with support for different search types:
    - similarity: Standard similarity search returning documents
    - mmr: Maximal Marginal Relevance for diversity
    """
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the vector store retriever.
        
        Args:
            vector_store: The vector store to retrieve from
        """
        self.vector_store = vector_store
        logger.info(f"Initialized VectorStoreRetriever with {type(vector_store).__name__}")

    async def retrieve_similarity(
        self, query: str, k: int = 4, sim_threshold: float = float("inf"), filters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """Retrieve documents based on vector similarity.
        
        Args:
            query: The search query text
            k: Number of documents to retrieve (default: 4)
            sim_threshold: Minimum similarity score threshold. Documents with scores
                below this threshold will be filtered out. Use float("inf") to disable
                filtering (default: float("inf"))
            filters: Optional metadata filters to narrow down the search
            **kwargs: Additional keyword arguments passed to the vector store
            
        Returns:
            List of Document objects ordered by similarity score
        """
        search_kwargs = {"filters": filters, **kwargs} if filters else kwargs

        if sim_threshold != float("inf"):
            docs_with_scores = await self.vector_store.similarity_search_with_score(
                query=query, k=k, **search_kwargs
            )
            return [doc for doc, score in docs_with_scores if score >= sim_threshold]

        return await self.vector_store.similarity_search(
            query=query, k=k, **search_kwargs
        )

    async def retrieve_mmr(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """Retrieve documents using Maximal Marginal Relevance for diversity.
        
        MMR optimizes for both similarity to query AND diversity among selected documents,
        reducing redundancy in the results.
        
        Args:
            query: The search query text
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch for MMR algorithm (default: 20).
                The algorithm will first fetch this many documents, then select k
                diverse documents from them
            lambda_mult: Balance between similarity (1.0) and diversity (0.0). 
                Default: 0.5 for balanced results
            filters: Optional metadata filters to narrow down the search
            **kwargs: Additional keyword arguments passed to the vector store
            
        Returns:
            List of Document objects selected by maximal marginal relevance
            
        Raises:
            NotImplementedError: If the underlying vector store doesn't support MMR
        """
        if not hasattr(self.vector_store, "max_marginal_relevance_search"):
            raise NotImplementedError("MMR search is not supported by the underlying vector store.")
        
        search_kwargs = {"filters": filters, **kwargs} if filters else kwargs
        return await self.vector_store.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **search_kwargs
        )

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: SearchType = "similarity",
        **kwargs
    ) -> List[Document]:
        """Generic retrieval method that dispatches to specific search methods.
        
        Args:
            query: The search query text
            k: Number of documents to retrieve (default: 4)
            search_type: Type of search to perform. Options:
                - "similarity": Standard similarity search (default)
                - "mmr": Maximal Marginal Relevance for diverse results
            **kwargs: Additional arguments passed to the specific retrieval method.
                For similarity: sim_threshold, filters
                For mmr: fetch_k, lambda_mult, filters
                
        Returns:
            List of Document objects ordered by relevance
            
        Note:
            For a more explicit and type-safe API, prefer using the specific methods:
            - retrieve_similarity() for standard similarity search
            - retrieve_mmr() for diverse results with MMR
        """
        logger.debug(f"Dispatching retrieval for search_type='{search_type}'")
        if search_type == "similarity":
            return await self.retrieve_similarity(query, k, **kwargs)

        if search_type == "mmr":
            return await self.retrieve_mmr(query, k, **kwargs)
        
        raise ValueError(f"Invalid search_type: {search_type}")