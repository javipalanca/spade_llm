"""Base classes for text splitters."""

import copy
import logging
from typing import List, Optional, Callable, Dict, Any, Literal
from abc import ABC, abstractmethod

from ..core.document import Document

logger = logging.getLogger(__name__)


class TextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    chunk_size: int
    chunk_overlap: int
    length_function: Callable[[str], int]
    keep_separator: bool | Literal["start", "end"]
    add_start_index: bool
    strip_whitespace: bool

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,
        add_start_index: bool = True,
        strip_whitespace: bool = True,
    ):
        """Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it.
                           Can be False, True (equivalent to "start"), "start", or "end"
            add_start_index: If True, includes chunk's start index in metadata
            strip_whitespace: If True, strips whitespace from the start and end of chunks
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator
        self.add_start_index = add_start_index
        self.strip_whitespace = strip_whitespace
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunk strings
        """

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create Document objects from a list of texts with optional metadata.
        
        Splits each text and creates Document objects with metadata tracking.
        
        Args:
            texts: List of text strings to convert to documents
            metadatas: Optional list of metadata dictionaries (one per text)
            
        Returns:
            List of Document objects created from the split texts
        """
        metadatas = metadatas or [{}] * len(texts)
        documents = []
        
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            chunk_index = 0
            
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas[i])
                
                # Track start index if requested
                if self.add_start_index:
                    offset = index + previous_chunk_len - self.chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                
                # Add chunk index to metadata
                metadata["chunk_index"] = chunk_index
                chunk_index += 1
                
                new_doc = Document(content=chunk, metadata=metadata)
                documents.append(new_doc)
        
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split Document objects into smaller Document chunks.
        
        Each input document is split into multiple chunks, with metadata
        preserved and enhanced with chunking information.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of Document objects representing chunks with metadata
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.content)
            metadata = copy.deepcopy(doc.metadata)
            metadatas.append(metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """Join document chunks with a separator.
        
        Args:
            docs: List of document chunks to join
            separator: Separator to use between chunks
            
        Returns:
            Joined text or None if result is empty
        """
        text = separator.join(docs)
        if self.strip_whitespace:
            text = text.strip()
        return text if text else None

    def _merge_splits(self, splits: List[str], separator: str = "") -> List[str]:
        """Merge splits into chunks of appropriate size with overlap.
        
        Inspired by LangChain's approach: combines smaller pieces into medium-sized
        chunks while respecting size limits and maintaining overlap between chunks.
        
        Args:
            splits: List of text splits to merge
            separator: Separator to use when joining splits
            
        Returns:
            List of merged text chunks
        """
        separator_len = self.length_function(separator)
        
        docs = []
        current_doc: List[str] = []
        total = 0
        
        for split in splits:
            split_len = self.length_function(split)
            
            # Check if adding this split would exceed chunk size
            potential_size = total + split_len + (separator_len if current_doc else 0)
            
            if potential_size > self.chunk_size:
                # Warn if current chunk exceeds target size
                if total > self.chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, which is longer than "
                        f"the specified {self.chunk_size}"
                    )
                
                # Save current chunk if not empty
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    
                    # Apply overlap by keeping some splits from the end
                    while total > self.chunk_overlap or (
                        total + split_len + (separator_len if current_doc else 0) > self.chunk_size
                        and total > 0
                    ):
                        removed_len = self.length_function(current_doc[0])
                        total -= removed_len + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            
            # Add split to current chunk
            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)
        
        # Add the last chunk
        if current_doc:
            doc = self._join_docs(current_doc, separator)
            if doc is not None:
                docs.append(doc)
        
        return docs
