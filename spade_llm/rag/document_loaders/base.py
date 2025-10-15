"""Base classes for document loaders."""

import logging
from typing import List, AsyncGenerator, Set
from abc import ABC, abstractmethod

from ..core.document import Document

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    async def load_stream(self) -> AsyncGenerator[Document, None]:
        """Stream documents from the source as an async generator.
        
        Yields:
            Document objects as they become available
        """
        # This is a placeholder
        if False:
            yield

    async def load(self) -> List[Document]:
        """Load all documents from the source into a list.

        This method provides a default implementation by consuming the
        `load_stream` generator. Subclasses that can load documents
        more efficiently in a single batch may override this method.

        Returns:
            A list of all loaded Document objects
        """
        return [doc async for doc in self.load_stream()]

    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """Get the set of file extensions supported by this loader.

        Returns:
            Set of file extension strings (e.g., {'.txt', '.md'})
        """
        return set()

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if this loader supports the given file extension.

        Args:
            extension: File extension to check (with or without the dot)

        Returns:
            True if the extension is supported, False otherwise
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        return extension.lower() in cls.get_supported_extensions()
