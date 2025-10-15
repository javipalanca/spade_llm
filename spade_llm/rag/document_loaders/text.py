"""Text file document loaders."""

import inspect
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Set, Type

import aiofiles

from .base import BaseDocumentLoader
from ..core.document import Document

logger = logging.getLogger(__name__)


class TextLoader(BaseDocumentLoader):
    """Loads a single text file as one Document.

    Supports common text formats like .txt, .md, and .rst.
    """

    SUPPORTED_EXTENSIONS: Set[str] = {'.txt', '.md', '.rst'}

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the text file loader.

        Args:
            file_path: Path to the text file
            encoding: File encoding to use
            metadata: Additional metadata to attach to the document
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.metadata = metadata or {}

    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        return cls.SUPPORTED_EXTENSIONS.copy()

    async def load_stream(self) -> AsyncGenerator[Document, None]:
        """Reads the file and yields a single Document.
        
        Yields:
            Document object containing the file's content
        """
        try:
            async with aiofiles.open(
                self.file_path, mode="r", encoding=self.encoding
            ) as f:
                content = await f.read()

            metadata = {
                **self.metadata,
                "source": self.file_path,
            }
            yield Document(content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {e}")
            raise


class DirectoryLoader(BaseDocumentLoader):
    """Loads all supported documents from a directory.

    Recursively finds files and uses the appropriate registered loader
    for each file type.
    """

    DEFAULT_LOADER_MAP: Dict[str, Type[BaseDocumentLoader]] = {
        ext: TextLoader for ext in TextLoader.SUPPORTED_EXTENSIONS
    }

    def __init__(
        self,
        path: Union[str, Path],
        glob_pattern: str = "**/*",
        recursive: bool = True,
        suffixes: Optional[List[str]] = None,
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, Any]] = None,
        loader_map: Optional[Dict[str, Type[BaseDocumentLoader]]] = None,
    ):
        """Initializes the directory loader.

        Args:
            path: Directory path to load from
            glob_pattern: Glob pattern to match files against
            recursive: If True, search subdirectories. Overridden by glob_pattern
            suffixes: List of file suffixes to include (e.g., ['.txt', '.md']).
                      If None, uses all extensions from the loader_map
            encoding: Default file encoding for text-based loaders
            metadata: Base metadata to attach to all loaded documents
            loader_map: A map of file extensions to loader classes
        """
        self.path = Path(path)
        self.glob_pattern = glob_pattern if recursive else "*"
        self.encoding = encoding
        self.metadata = metadata or {}
        self.loader_map = loader_map or self.DEFAULT_LOADER_MAP.copy()

        if suffixes is not None:
            self.suffixes = {s if s.startswith('.') else f'.{s}' for s in suffixes}
        else:
            self.suffixes = set(self.loader_map.keys())

    def _get_loader_for_file(self, file_path: Path) -> Optional[BaseDocumentLoader]:
        """Instantiates the appropriate loader for a given file path.
        
        Args:
            file_path: The path to the file to find a loader for

        Returns:
            An initialized document loader instance if a suitable one is
            found, otherwise None
        """
        extension = file_path.suffix.lower()
        if extension not in self.suffixes:
            return None
        
        loader_class = self.loader_map.get(extension)
        if loader_class is None:
            return None

        # Inspect the loader's constructor to find its supported parameters
        init_signature = inspect.signature(loader_class.__init__)
        accepted_params = set(init_signature.parameters.keys())

        # Define all arguments this DirectoryLoader could potentially pass
        potential_kwargs = {
            'file_path': file_path,
            'encoding': self.encoding,
            'metadata': self.metadata
        }
        
        # Filter for arguments that the specific loader class actually accepts
        kwargs_to_pass = {
            k: v for k, v in potential_kwargs.items() if k in accepted_params
        }
        
        return loader_class(**kwargs_to_pass)

    async def load_stream(self) -> AsyncGenerator[Document, None]:
        """Finds and streams all supported files in the directory.
        
        Yields:
            Document object loaded from a file in the directory
        """
        # Check if directory exists first
        if not self.path.is_dir():
            if not self.path.exists():
                raise FileNotFoundError(f"Directory not found: {self.path}")
            raise NotADirectoryError(f"Path is not a directory: {self.path}")
        
        try:
            # Generator expression to process file paths lazily
            file_paths = (
                p for p in self.path.glob(self.glob_pattern)
                if p.is_file() and p.suffix.lower() in self.suffixes
            )
        except Exception as e:
            logger.error(f"Error accessing directory {self.path}: {e}")
            raise

        for file_path in file_paths:
            try:
                loader = self._get_loader_for_file(file_path)
                if loader:
                    async for doc in loader.load_stream():
                        yield doc
            except Exception as e:
                # Log the error for the specific file and continue with the rest
                logger.warning(f"Failed to load {file_path}: {e}")