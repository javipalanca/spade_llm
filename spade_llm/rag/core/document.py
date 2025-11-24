"""Core document representation for the RAG system."""

from dataclasses import dataclass, field
from typing import Dict, Any, TypeVar
from uuid import uuid4

# A generic type for the class itself, used in from_dict
TDocument = TypeVar("TDocument", bound="Document")


@dataclass
class Document:
    """Class representing document: a single piece of text and its metadata.

    Attributes:
        content: The main text content of the document
        metadata: A dictionary of additional information
            about the document (e.g., source, page number)
        id: Unique identifier for the document (auto-generated UUID)
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        """Validate document attributes after initialization."""
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.id, str):
            raise TypeError("id must be a string")
        if not self.id or not self.id.strip():
            raise ValueError("id must be a non-empty string")

    @property
    def text(self) -> str:
        """Alias for the 'content' attribute for backward compatibility."""
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Document instance to a dictionary.

        Returns:
            A dictionary representation of the document
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.copy()
        }

    @classmethod
    def from_dict(cls: type[TDocument], data: Dict[str, Any]) -> TDocument:
        """Creates a Document instance from a dictionary.

        Args:
            data: A dictionary with 'content' and optional 'id' and 'metadata' keys.
                If 'id' is not provided, a new UUID will be generated automatically

        Returns:
            A new instance of the Document class
        """
        if "content" not in data:
            raise KeyError("The 'content' key is required to create a Document.")

        return cls(
            content=data["content"],
            id=data.get("id", str(uuid4())),
            metadata=data.get("metadata", {})
        )