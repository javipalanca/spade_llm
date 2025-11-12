"""Tests for retrieval utility functions."""

import json
import pytest

from spade_llm.utils.retrieval_utils import (
    format_documents_for_response,
    create_retrieval_response_body,
)
from spade_llm.rag.core.document import Document


# Fixtures
@pytest.fixture
def sample_docs():
    """Provides a list of sample Document objects."""
    doc1 = Document(content="Document 1", metadata={"source": "test1.txt"})
    doc2 = Document(content="Document 2", metadata={"source": "test2.txt"})
    return [doc1, doc2]


class TestFormatDocumentsForResponse:
    """Test format_documents_for_response function."""

    def test_format_documents_without_scores(self, sample_docs):
        """Test formatting documents without scores."""
        formatted = format_documents_for_response(sample_docs)

        assert len(formatted) == 2
        assert formatted[0]["content"] == "Document 1"
        assert formatted[0]["metadata"] == {"source": "test1.txt"}
        assert "score" not in formatted[0]
        assert formatted[1]["content"] == "Document 2"
        assert formatted[1]["metadata"] == {"source": "test2.txt"}
        assert "score" not in formatted[1]

    def test_format_empty_list(self):
        """Test formatting empty document list."""
        formatted = format_documents_for_response([])
        assert formatted == []

    def test_format_documents_with_empty_metadata(self):
        """Test formatting documents with empty metadata."""
        doc = Document(content="Test content", metadata={})
        results = [doc]

        formatted = format_documents_for_response(results)

        assert len(formatted) == 1
        assert formatted[0]["content"] == "Test content"
        assert formatted[0]["metadata"] == {}

    def test_format_documents_with_unicode_content(self):
        """Test formatting documents with unicode content."""
        doc = Document(content="Documento en español 🇪🇸", metadata={"lang": "es"})
        results = [doc]

        formatted = format_documents_for_response(results)

        assert formatted[0]["content"] == "Documento en español 🇪🇸"
        assert formatted[0]["metadata"] == {"lang": "es"}

    def test_format_documents_with_complex_metadata(self):
        """Test formatting documents with complex metadata."""
        doc = Document(
            content="Test",
            metadata={
                "source": "file.txt",
                "author": "John Doe",
                "tags": ["tag1", "tag2"],
                "nested": {"key": "value"},
            },
        )
        results = [doc]

        formatted = format_documents_for_response(results)

        assert formatted[0]["metadata"]["source"] == "file.txt"
        assert formatted[0]["metadata"]["author"] == "John Doe"
        assert formatted[0]["metadata"]["tags"] == ["tag1", "tag2"]
        assert formatted[0]["metadata"]["nested"] == {"key": "value"}
    
    def test_format_documents_raises_on_malformed_input(self):
        """Test that format_documents_for_response raises errors on malformed input."""
        # Non-Document item
        with pytest.raises(TypeError, match="Expected Document, got str"):
            format_documents_for_response(["just a string"])  # type: ignore


class TestCreateRetrievalResponseBody:
    """Test create_retrieval_response_body function."""

    def test_create_response_without_scores(self, sample_docs):
        """Test creating JSON response body without scores."""
        response_body = create_retrieval_response_body(sample_docs)
        parsed = json.loads(response_body)

        assert isinstance(response_body, str)
        assert "documents" in parsed
        assert len(parsed["documents"]) == 2
        assert parsed["documents"][0]["content"] == "Document 1"
        assert parsed["documents"][1]["content"] == "Document 2"
        assert "score" not in parsed["documents"][0]
        assert "score" not in parsed["documents"][1]

    def test_create_response_empty_results(self):
        """Test creating response with empty results."""
        response_body = create_retrieval_response_body([])
        parsed_response = json.loads(response_body)

        expected = {"documents": []}
        assert parsed_response == expected

    def test_create_response_is_valid_json(self):
        """Test that the created response is valid JSON."""
        doc = Document(content="Test", metadata={})
        results = [doc]

        response_body = create_retrieval_response_body(results)

        # Should not raise exception
        parsed = json.loads(response_body)
        assert "documents" in parsed
        assert len(parsed["documents"]) == 1

    def test_create_response_with_special_characters(self):
        """Test creating response with special characters in content."""
        doc = Document(
            content='Test "quoted" content with \n newlines',
            metadata={"key": "value with 'quotes'"},
        )
        results = [doc]

        response_body = create_retrieval_response_body(results)

        # Should be valid JSON
        parsed = json.loads(response_body)
        assert parsed["documents"][0]["content"] == 'Test "quoted" content with \n newlines'

    def test_create_response_formatting(self):
        """Test that response is formatted with indentation."""
        doc = Document(content="Test", metadata={})
        results = [doc]

        response_body = create_retrieval_response_body(results)

        expected_data = {"documents": [{"content": "Test", "metadata": {}}]}
        expected_json_string = json.dumps(expected_data, indent=2)

        assert response_body == expected_json_string


class TestRetrievalUtilsIntegration:
    """Integration tests for retrieval utility functions."""

    def test_full_workflow_without_scores(self, sample_docs):
        """Test complete workflow from documents to JSON response."""
        # Format documents
        formatted = format_documents_for_response(sample_docs)
        assert len(formatted) == 2

        # Create response body
        response_body = create_retrieval_response_body(sample_docs)
        parsed = json.loads(response_body)

        assert len(parsed["documents"]) == 2
        assert parsed["documents"][0]["content"] == "Document 1"
        assert parsed["documents"][1]["content"] == "Document 2"

    def test_consistency_between_format_and_create(self):
        """Test that format_documents and create_response are consistent."""
        doc = Document(content="Test", metadata={"key": "value"})
        results = [doc]

        # Get formatted documents
        formatted = format_documents_for_response(results)

        # Get response body and parse it
        response_body = create_retrieval_response_body(results)
        parsed = json.loads(response_body)

        # They should contain the same document data
        assert formatted == parsed["documents"]
