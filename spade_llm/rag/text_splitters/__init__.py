"""Text splitters for document chunking.

This module provides various text splitting strategies for breaking down large
documents into smaller chunks suitable for embedding and vector storage.
"""

from .base import TextSplitter
from .character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

__all__ = [
    "TextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter"
]