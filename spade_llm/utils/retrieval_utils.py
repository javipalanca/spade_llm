"""Utility functions for retrieval operations."""

import json
from typing import Any, Dict, List
from ..rag.core.document import Document

def format_documents_for_response(results: List[Document]) -> List[Dict[str, Any]]:
    """
    Format retrieved documents for a response message.

    Args:
        results: List of retrieved documents

    Returns:
        List of formatted document dictionaries with content and metadata
    """
    formatted = []
    
    for doc in results:
        if not isinstance(doc, Document):
            raise TypeError(f"Expected Document, got {type(doc).__name__}")

        entry = {"content": doc.content, "metadata": doc.metadata}
        
        formatted.append(entry)
    
    return formatted


def create_retrieval_response_body(results: List[Document]) -> str:
    """Create a JSON response body containing formatted documents."""
    documents = format_documents_for_response(results)
    return json.dumps({"documents": documents}, indent=2)
