"""ChromaDB vector store implementation.

Provides async operations for ChromaDB vector store with support for:
- Similarity search and maximal marginal relevance (MMR) search
- Document lifecycle management (add, update, delete)
- Direct retrieval by IDs
- Flexible client configuration (persistent, HTTP, or custom client)
- Collection management and metadata filtering
"""

import logging
from typing import List, Optional, Tuple, Any, Callable
import json
import asyncio
import uuid as uuid_module

from .base import VectorStore
from ..core.document import Document

from chromadb.config import Settings
from chromadb.api.collection_configuration import CreateCollectionConfiguration

logger = logging.getLogger(__name__)
DEFAULT_K = 4  # Number of Documents to return


def _maximal_marginal_relevance(
    query_embedding: Any,  # np.ndarray
    embedding_list: Any,  # np.ndarray
    k: int = 4,
    lambda_mult: float = 0.5,
) -> List[int]:
    """Calculate maximal marginal relevance indices.
    
    Args:
        query_embedding: Query embedding
        embedding_list: Array of embeddings to select from
        k: Number of documents to return
        lambda_mult: Balance between similarity and diversity
            
    Returns:
        List of indices of embeddings selected by maximal marginal relevance
    """
    import numpy as np
    
    if min(k, len(embedding_list)) <= 0:
        return []
    
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Cosine similarity
    similarity_to_query = _cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = _cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    
    return idxs


def _cosine_similarity(X: Any, Y: Any) -> Any:  # np.ndarray types
    """Row-wise cosine similarity between matrices X and Y."""
    import numpy as np
    
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. "
            f"X has shape {X.shape} and Y has shape {Y.shape}."
        )
    
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    
    return similarity


def _euclidean_relevance_score_fn(distance: float) -> float:
    """Convert Euclidean distance to a similarity score on a scale [0, 1]."""
    return 1.0 / (1.0 + distance)


def _max_inner_product_relevance_score_fn(distance: float) -> float:
    """Convert max inner product distance to similarity score.
    
    ChromaDB returns the negative of the inner product as the distance,
    so we negate it to get the actual inner product.
    """
    return -distance


def _cosine_relevance_score_fn(distance: float) -> float:
    """Convert cosine distance to similarity score."""
    return 1.0 - distance


class Chroma(VectorStore):
    """ChromaDB vector store with async operations."""
    
    MAX_RESULTS = 1000
    _DEFAULT_COLLECTION_NAME = "documents"
    
    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ssl: bool = False,
        headers: Optional[dict] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        embedding_fn: Optional[Callable] = None,
        collection_metadata: Optional[dict] = None,
        collection_configuration: Optional[CreateCollectionConfiguration] = None,
        client_settings: Optional[Settings] = None,
        client: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        """Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (for persistent client)
            host: Host for ChromaDB server (for HTTP client)
            port: Port for ChromaDB server (for HTTP client). Default is 8000
            ssl: Whether to use SSL for HTTP client. Default is False
            headers: Optional HTTP headers for HTTP client
            tenant: Tenant ID. Default is 'default_tenant'
            database: Database name. Default is 'default_database'
            embedding_fn: Optional async function that takes a list of texts and returns
                their embeddings. Example: provider.get_embeddings
            collection_metadata: Optional metadata for the collection
            collection_configuration: Optional configuration for the collection.
                Use chromadb.api.collection_configuration.CreateCollectionConfiguration
                to define index properties (e.g., distance function).
                Example: CreateCollectionConfiguration(hnsw={"space": "cosine"})
            client_settings: Optional Chroma client settings (chromadb.config.Settings)
            client: Optional pre-configured Chroma client
            relevance_score_fn: Optional function to convert distance to relevance score.
                If None, will be auto-selected based on collection's distance function
        """
        super().__init__(embedding_fn=embedding_fn)
        
        # Validate that only one connection method is provided
        connection_methods = sum([
            persist_directory is not None,
            host is not None,
            client is not None,
        ])
        if connection_methods > 1:
            raise ValueError(
                "You can only specify one of: persist_directory, host, or client. "
                "Please choose a single connection method."
            )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.host = host
        self.port = port or 8000
        self.ssl = ssl
        self.headers = headers
        self.tenant = tenant
        self.database = database
        self.collection_metadata = collection_metadata
        self.collection_configuration = collection_configuration
        self.client_settings = client_settings
        self._provided_client = client
        self._client = None
        self._collection = None
        self._init_lock = asyncio.Lock()
        self._relevance_score_fn = relevance_score_fn

    async def __aenter__(self):
        """Enter async context manager, initializing the store."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager, cleaning up resources."""
        await self.cleanup()

    async def _ensure_initialized(self):
        """Ensure the client and collection are initialized, with thread safety."""
        if self._collection is None:
            async with self._init_lock:
                if self._collection is None:
                    await self.initialize()

    def _serialize_metadata(self, metadata: dict) -> dict:
        """Serialize metadata to ensure it's JSON-compatible.
        
        Args:
            metadata: Original metadata dictionary
        
        Returns:
            JSON-serializable metadata dictionary
        """
        serializable_metadata = {}
        for k, v in metadata.items():
            try:
                json.dumps(v)
                serializable_metadata[k] = v
            except (TypeError, ValueError):
                serializable_metadata[k] = str(v)
        return serializable_metadata

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select relevance score function based on collection's distance function.
        
        The most similar documents will have the lowest relevance score. Default
        relevance score function is Euclidean distance. Distance metric must be
        provided in `collection_configuration` during initialization.
        Example: collection_configuration=CreateCollectionConfiguration(hnsw={"space": "cosine"}).
        Available distance metrics: 'cosine', 'l2', 'ip'.
        
        Returns:
            Function that converts distance to relevance score
        """
        if self._relevance_score_fn is not None:
            return self._relevance_score_fn
        
        if self.collection_configuration is not None:
            try:
                # Access configuration using attribute access for CreateCollectionConfiguration
                hnsw_config = getattr(self.collection_configuration, 'hnsw', None)
                hnsw_space = hnsw_config.get("space") if isinstance(hnsw_config, dict) else None
                
                # Check for SPANN configuration (alternative indexing method)
                spann_config = getattr(self.collection_configuration, 'spann', None)
                spann_space = spann_config.get("space") if isinstance(spann_config, dict) else None
                
                space = hnsw_space or spann_space
                
                if space == "cosine":
                    return _cosine_relevance_score_fn
                elif space == "ip":
                    return _max_inner_product_relevance_score_fn
                elif space == "l2":
                    return _euclidean_relevance_score_fn
            except Exception as e:
                logger.warning(f"Could not determine distance function from collection configuration: {e}")
        
        # Default to L2/Euclidean (Chroma's default)
        return _euclidean_relevance_score_fn

    def _create_documents_from_results(
        self, 
        ids: List[str], 
        documents: List[str], 
        metadatas: List[Any], 
        distances: Optional[List[float]] = None
    ) -> List:
        """Create Document objects from ChromaDB query results.
        
        Args:
            ids: List of document IDs
            documents: List of document contents
            metadatas: List of metadata dictionaries
            distances: Optional list of distances from the query
            
        Returns:
            List of Document objects, or list of (Document, score) tuples if distances provided
        """
        docs = []
        for i, (doc_id, content, meta) in enumerate(zip(ids, documents, metadatas)):
            if not isinstance(meta, dict):
                meta = {}
            doc = Document(id=doc_id, content=content, metadata=meta)
            # Convert distance to relevance score using the appropriate function
            if distances is not None:
                relevance_fn = self._select_relevance_score_fn()
                score = relevance_fn(distances[i])
                docs.append((doc, score))
            else:
                docs.append(doc)
        return docs

    async def initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        if self._collection is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB is required for Chroma. "
                "Install with: pip install chromadb"
            )

        # Use provided client if available
        if self._provided_client is not None:
            self._client = self._provided_client
            logger.info("Using provided ChromaDB client")
        else:
            # Default values for tenant and database
            _tenant = self.tenant if self.tenant else "default_tenant"
            _database = self.database if self.database else "default_database"
            _settings = self.client_settings or Settings(anonymized_telemetry=False)

            # Create client based on configuration
            if self.host:
                # HTTP client for server mode
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    ssl=self.ssl,
                    headers=self.headers,
                    settings=_settings,
                    tenant=_tenant,
                    database=_database,
                )
                logger.info(f"Initialized ChromaDB HTTP client: {self.host}:{self.port}")
            elif self.persist_directory:
                # Persistent client
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=_settings,
                    tenant=_tenant,
                    database=_database,
                )
                logger.info(f"Initialized ChromaDB persistent client: {self.persist_directory}")
            else:
                # In-memory client
                self._client = chromadb.Client(settings=_settings)
                logger.info("Initialized ChromaDB in-memory client")

        # Create collection with proper parameters
        collection_kwargs = {
            "name": self.collection_name,
            "embedding_function": None,  # We handle embeddings ourselves
        }
        
        if self.collection_metadata is not None:
            collection_kwargs["metadata"] = self.collection_metadata
        
        if self.collection_configuration is not None:
            collection_kwargs["configuration"] = self.collection_configuration
        
        self._collection = self._client.get_or_create_collection(**collection_kwargs)
        logger.info(f"Using ChromaDB collection: {self.collection_name}")

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Store documents with their embeddings.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs that were added
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        if not documents:
            return []
        if not self.embedding_fn:
            raise ValueError("No embedding function available for document embedding")
        
        texts = [doc.content for doc in documents]
        embeddings_list = await self.embedding_fn(texts)
        ids = [doc.id for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [self._serialize_metadata(doc.metadata) if doc.metadata else None for doc in documents]
        
        await asyncio.to_thread(
            self._collection.upsert,
            ids=ids,
            documents=documents_text,
            embeddings=embeddings_list, # type: ignore
            metadatas=metadatas # type: ignore
        )
        
        logger.debug(f"Stored {len(documents)} documents in ChromaDB")
        return ids

    async def _search(self, query: str, k: int, **kwargs):
        """Private helper to run the core query and return raw results."""
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        if not self.embedding_fn:
            raise ValueError("No embedding function available for query embedding")
        
        # For single query, wrap in list and unwrap result
        query_embeddings = await self.embedding_fn([query])
        query_embedding = query_embeddings[0]
        return await self._search_by_vector(query_embedding, k, **kwargs)

    async def _search_by_vector(self, embedding: List[float], k: int, **kwargs):
        """Private helper to run query by vector and return raw results."""
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        
        where_clause = kwargs.get('filters') or kwargs.get('where')
        where_document_clause = kwargs.get('where_document')
        
        query_kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(k, self.MAX_RESULTS),
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where_clause is not None:
            query_kwargs["where"] = where_clause
        if where_document_clause is not None:
            query_kwargs["where_document"] = where_document_clause
        
        results = await asyncio.to_thread(
            self._collection.query,
            **query_kwargs
        )
        return results

    async def similarity_search(self, query: str, k: int = DEFAULT_K, **kwargs) -> List[Document]:
        """Perform similarity search against the vector store.
        
        Args:
            query: Search query text
            k: Maximum number of results to return
            **kwargs: Additional search parameters (filters, where)
            
        Returns:
            List of Document objects ordered by similarity score
        """
        results = await self._search(query, k, **kwargs)
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        return self._create_documents_from_results(ids, docs, metadatas)

    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = DEFAULT_K, 
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores.
        
        Args:
            query: Search query text
            k: Maximum number of results to return
            **kwargs: Additional search parameters (filters, where)
            
        Returns:
            List of (Document, score) tuples ordered by similarity score
        """
        results = await self._search(query, k, **kwargs)
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return self._create_documents_from_results(ids, docs, metadatas, distances)

    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[dict] = None,
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
            filters: Filter by metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            List of documents selected by maximal marginal relevance
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        if not self.embedding_fn:
            raise ValueError("No embedding function available for query embedding")
        
        # For single query, wrap in list and unwrap result
        query_embeddings = await self.embedding_fn([query])
        query_embedding = query_embeddings[0]
        
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for MMR search. Install with: pip install numpy")
        
        # Fetch more results than needed for MMR algorithm
        where_clause = filters or kwargs.get('where')
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, self.MAX_RESULTS),
            where=where_clause,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        # Extract data
        embeddings = (results.get("embeddings") or [[]])[0]
        if len(embeddings) == 0:
            return []
        
        # Calculate MMR
        mmr_selected = _maximal_marginal_relevance(
            np.array(query_embedding, dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        
        # Get the selected documents
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        
        candidates = self._create_documents_from_results(ids, docs, metadatas)
        
        return [candidates[i] for i in mmr_selected if i < len(candidates)]


    async def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of Document objects
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        
        results = await asyncio.to_thread(
            self._collection.get,
            ids=ids,
            include=["documents", "metadatas"]
        )
        
        doc_ids = results.get("ids") or []
        docs = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        
        return self._create_documents_from_results(doc_ids, docs, metadatas)

    async def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[dict] = None,
        include: Optional[List[str]] = None,
    ) -> dict:
        """Get items from the collection with flexible filtering options.
        
        Args:
            ids: Document IDs to retrieve. Optional
            where: Filter results by metadata.
                E.g. {"color": "red", "price": {"$lt": 5.0}}. Optional
            limit: Number of documents to return. Optional
            offset: Offset to start returning results from.
                Useful for paging results with limit. Optional
            where_document: Filter by document content.
                E.g. {"$contains": "hello"}. Optional
            include: What to include in the results.
                Can contain "embeddings", "metadatas", "documents".
                IDs are always included.
                Defaults to ["metadatas", "documents"]. Optional
                
        Returns:
            Dictionary with keys "ids", "embeddings", "metadatas", "documents"
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        
        kwargs: dict = {}
        if ids is not None:
            kwargs["ids"] = ids
        if where is not None:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if where_document is not None:
            kwargs["where_document"] = where_document
        if include is not None:
            kwargs["include"] = include
        
        results = await asyncio.to_thread(
            self._collection.get,
            **kwargs
        )
        
        return results  # type: ignore

    async def update_document(self, document_id: str, document: Document) -> None:
        """Update a single document in the collection.
        
        Args:
            document_id: ID of the document to update.
            document: New document content.
        """
        await self.update_documents([document_id], [document])

    async def update_documents(self, ids: List[str], documents: List[Document]) -> None:
        """Update multiple documents in the collection.
        
        Args:
            ids: List of document IDs to update.
            documents: List of new document contents.
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        if not self.embedding_fn:
            raise ValueError("No embedding function available for document embedding")
        
        texts = [doc.content for doc in documents]
        metadatas = [self._serialize_metadata(doc.metadata) if doc.metadata else None for doc in documents]
        embeddings_list = await self.embedding_fn(texts)
        
        try:
            import numpy as np
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
        except ImportError:
            raise ImportError("numpy is required for embedding update in Chroma. Install with: pip install numpy")
        
        # Convert to list for better compatibility
        embeddings_list_final = embeddings_array.tolist()
        
        await asyncio.to_thread(
            self._collection.update,
            ids=ids,
            embeddings=embeddings_list_final,
            documents=texts,
            metadatas=metadatas  # type: ignore
        )
        
        logger.debug(f"Updated {len(documents)} documents in ChromaDB")

    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        try:
            await asyncio.to_thread(self._collection.delete, ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    async def get_document_count(self) -> int:
        """Get the total number of documents in the store.
        
        Returns:
            Number of documents in the store
        """
        await self._ensure_initialized()
        if not self._collection:
            raise RuntimeError("Chroma could not be initialized")
        count = await asyncio.to_thread(self._collection.count)
        return count

    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        await self._ensure_initialized()
        if not self._client:
            raise RuntimeError("Chroma client not initialized")
        
        if self._collection:
            await asyncio.to_thread(
                self._client.delete_collection,
                name=self.collection_name
            )
            logger.info(f"Deleted collection: {self.collection_name}")
            self._collection = None

    async def reset_collection(self) -> None:
        """Reset the collection by deleting and recreating it.
        
        Useful for testing or when you want to completely clear and reinitialize
        a collection.
        """
        await self._ensure_initialized()
        if not self._client:
            raise RuntimeError("Chroma client not initialized")
        
        if self._collection:
            await self.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
        
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self.collection_name,
            metadata=self.collection_metadata,
        )
        logger.info(f"Recreated collection: {self.collection_name}")

    @classmethod
    async def from_documents(
        cls,
        documents: List[Document],
        embedding_fn: Optional[Callable] = None,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        **kwargs
    ) -> "Chroma":
        """Create a Chroma vector store from a list of documents.
        
        Args:
            documents: List of Document objects to add.
            embedding_fn: Async function that takes a list of texts and returns embeddings.
            collection_name: Name of the collection to create.
            **kwargs: Additional arguments to pass to Chroma constructor.
            
        Returns:
            Initialized Chroma vector store with documents added.
        """
        store = cls(
            collection_name=collection_name,
            embedding_fn=embedding_fn,
            **kwargs
        )
        await store.initialize()
        await store.add_documents(documents)
        return store

    @classmethod
    async def from_texts(
        cls,
        texts: List[str],
        embedding_fn: Optional[Callable] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        **kwargs
    ) -> "Chroma":
        """Create a Chroma vector store from a list of texts.
        
        Args:
            texts: List of text strings to add.
            embedding_fn: Async function that takes a list of texts and returns embeddings.
            metadatas: Optional list of metadata dicts for each text.
            ids: Optional list of IDs for each text.
            collection_name: Name of the collection to create.
            **kwargs: Additional arguments to pass to Chroma constructor.
            
        Returns:
            Initialized Chroma vector store with texts added.
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid_module.uuid4()) for _ in texts]
        
        # Generate empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) < len(texts):
            # Fill in missing metadata with empty dicts
            metadatas = metadatas + [{} for _ in range(len(texts) - len(metadatas))]
        
        # Create Document objects
        documents = [
            Document(id=id_, content=text, metadata=meta)
            for id_, text, meta in zip(ids, texts, metadatas)
        ]
        
        return await cls.from_documents(
            documents=documents,
            embedding_fn=embedding_fn,
            collection_name=collection_name,
            **kwargs
        )

    async def cleanup(self) -> None:
        """Clean up ChromaDB resources."""
        if self._collection:
            logger.info("Cleaning up ChromaDB collection and client")
        self._collection = None
        # Only clear client if we created it (not if it was provided)
        if not self._provided_client:
            self._client = None
