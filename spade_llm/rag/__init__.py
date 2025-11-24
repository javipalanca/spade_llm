"""RAG (Retrieval-Augmented Generation) implementation for SPADE LLM.

Components:
    - core: Core data structures (Document)
    - document_loaders: Load documents from various sources
    - text_splitters: Split documents into chunks
    - vector_stores: Store and search document embeddings
    - retrievers: Retrieve relevant documents for queries
"""

# Core document structure
from .core.document import Document

# Document loaders
from .document_loaders.base import BaseDocumentLoader
from .document_loaders.text import (
    TextLoader,
    DirectoryLoader
)

# Text splitters
from .text_splitters.base import TextSplitter
from .text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# Vector stores
from .vector_stores.base import VectorStore
try:
    from .vector_stores.chroma import Chroma
    _chroma_available = True
except ImportError:
    Chroma = None
    _chroma_available = False

# Retrievers
from .retrievers.base import BaseRetriever
from .retrievers.vector_store import VectorStoreRetriever

__all__ = [
    # Core
    "Document",
    
    # Document loaders
    "BaseDocumentLoader",
    "TextLoader", 
    "DirectoryLoader",
    
    # Text splitters
    "TextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    
    # Vector stores
    "VectorStore",
    
    # Retrievers
    "BaseRetriever",
    "VectorStoreRetriever",
]

# Add Chroma to __all__ only if it's available
if _chroma_available:
    __all__.append("Chroma")