"""Utility functions for retrieval operations."""

import json
from typing import Any, Dict, List

from ..rag.core.document import Document


def format_documents_for_response(
    results: List[Any], include_scores: bool = False
) -> List[Dict[str, Any]]:
    """
    Format retrieved documents for a response message.

    Args:
        results: Retrieved documents or (document, score) tuples
        include_scores: Whether results include scores

    Returns:
        List of formatted document dictionaries with content and metadata
    """
    formatted = []

    for item in results:
        if include_scores and isinstance(item, tuple):
            doc, score = item
            formatted.append(
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": round(score, 4),
                }
            )
        else:
            doc = item if isinstance(item, Document) else item[0]
            formatted.append({"content": doc.content, "metadata": doc.metadata})

    return formatted


def create_retrieval_response_body(
    results: List[Any], include_scores: bool = False
) -> str:
    """
    Create a JSON response body containing only the retrieved documents.

    Args:
        results: Retrieved documents or (document, score) tuples
        include_scores: Whether results include scores

    Returns:
        JSON string with documents array
    """
    documents = format_documents_for_response(results, include_scores)
    return json.dumps({"documents": documents}, indent=2)
