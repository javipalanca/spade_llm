"""Tests for the core Document class."""

import pytest
from spade_llm.rag.core.document import Document


class TestDocument:
    """Test cases for the Document class."""

    def test_initialization_minimal(self):
        """Test creating a document with minimal required fields."""
        doc = Document(content="Test content")
        assert isinstance(doc.id, str)
        assert len(doc.id) > 0
        assert doc.content == "Test content"
        assert doc.metadata == {}

    def test_initialization_with_metadata(self):
        """Test creating a document with metadata."""
        metadata = {"source": "test.txt", "author": "tester", "page": 1}
        doc = Document(content="Test content", metadata=metadata)
        assert doc.content == "Test content"
        assert isinstance(doc.id, str)
        assert len(doc.id) > 0
        assert doc.metadata == metadata
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["author"] == "tester"
        assert doc.metadata["page"] == 1

    def test_text_property_alias(self):
        """Test that text property is an alias for content."""
        doc = Document(content="Test content")
        assert doc.text == "Test content"
        assert doc.text == doc.content

    def test_invalid_content_type(self):
        """Test that non-string content raises TypeError."""
        with pytest.raises(TypeError, match="content must be a string"):
            Document(content=123)
        with pytest.raises(TypeError, match="content must be a string"):
            Document(content=None)
        with pytest.raises(TypeError, match="content must be a string"):
            Document(content=["list", "of", "strings"])

    def test_invalid_id_type(self):
        """Test that non-string id raises TypeError."""
        with pytest.raises(TypeError, match="id must be a string"):
            Document(content="Test", id=123)
        with pytest.raises(TypeError, match="id must be a string"):
            Document(content="Test", id=None)
    
    def test_empty_id(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            Document(content="Test", id="")
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            Document(content="Test", id="   ")

    def test_invalid_metadata_type(self):
        """Test that non-dict metadata raises TypeError."""
        with pytest.raises(TypeError, match="metadata must be a dictionary"):
            Document(content="Test", metadata="not a dict")
        with pytest.raises(TypeError, match="metadata must be a dictionary"):
            Document(content="Test", metadata=123)

    def test_to_dict(self):
        """Test converting document to dictionary."""
        metadata = {"source": "test.txt", "id": 1}
        doc = Document(content="Test content", id="test_doc_4", metadata=metadata)
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert doc_dict["id"] == "test_doc_4"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["metadata"] == metadata
        assert doc_dict["metadata"] is not doc.metadata  # Should be a copy

    def test_to_dict_empty_metadata(self):
        """Test to_dict with empty metadata."""
        doc = Document(content="Test content", id="test_doc_5")
        doc_dict = doc.to_dict()
        assert doc_dict["id"] == "test_doc_5"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["metadata"] == {}

    def test_from_dict(self):
        """Test creating document from dictionary."""
        data = {
            "id": "test_doc_6",
            "content": "Test content",
            "metadata": {"source": "test.txt", "id": 1}
        }
        doc = Document.from_dict(data)
        assert isinstance(doc, Document)
        assert doc.id == "test_doc_6"
        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test.txt", "id": 1}

    def test_from_dict_without_metadata(self):
        """Test from_dict when metadata is not provided."""
        data = {"id": "test_doc_7", "content": "Test content"}
        doc = Document.from_dict(data)
        assert doc.id == "test_doc_7"
        assert doc.content == "Test content"
        assert doc.metadata == {}

    def test_from_dict_missing_id(self):
        """Test that from_dict auto-generates id when missing."""
        data = {"content": "Test content"}
        doc = Document.from_dict(data)
        assert isinstance(doc.id, str)
        assert len(doc.id) > 0
        assert doc.content == "Test content"

    def test_round_trip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = Document(
            content="Test content",
            id="test_doc_8",
            metadata={"source": "test.txt", "author": "tester"}
        )
        doc_dict = original.to_dict()
        reconstructed = Document.from_dict(doc_dict)
        assert reconstructed.id == original.id
        assert reconstructed.content == original.content
        assert reconstructed.metadata == original.metadata

    def test_metadata_is_mutable(self):
        """Test that document metadata can be modified."""
        doc = Document(content="Test", id="test_doc_9", metadata={"key": "value"})
        doc.metadata["key"] = "new_value"
        doc.metadata["new_key"] = "new_value"
        assert doc.metadata["key"] == "new_value"
        assert doc.metadata["new_key"] == "new_value"

    def test_empty_string_content(self):
        """Test document with empty string content."""
        doc = Document(content="", id="test_doc_10")
        assert doc.id == "test_doc_10"
        assert doc.content == ""
        assert isinstance(doc.content, str)

    def test_large_content(self):
        """Test document with large content."""
        large_content = "A" * 100000
        doc = Document(content=large_content, id="test_doc_11")
        assert doc.id == "test_doc_11"
        assert len(doc.content) == 100000
        assert doc.content == large_content

    def test_complex_metadata(self):
        """Test document with complex metadata structures."""
        metadata = {
            "source": "test.txt",
            "tags": ["test", "document"],
            "stats": {"lines": 10, "words": 100},
            "nested": {"key": {"deep": "value"}}
        }
        doc = Document(content="Test", id="test_doc_12", metadata=metadata)
        assert doc.id == "test_doc_12"
        assert doc.metadata["tags"] == ["test", "document"]
        assert doc.metadata["stats"]["lines"] == 10
        assert doc.metadata["nested"]["key"]["deep"] == "value"

    def test_auto_generated_id_is_unique(self):
        """Test that auto-generated IDs are unique."""
        doc1 = Document(content="First document")
        doc2 = Document(content="Second document")
        assert doc1.id != doc2.id

    def test_document_equality(self):
        """Test the equality comparison of two documents."""
        doc1 = Document(content="Test", id="1", metadata={"source": "a"})
        doc2 = Document(content="Test", id="1", metadata={"source": "a"})
        assert doc1 == doc2

    def test_document_inequality(self):
        """Test the inequality of two documents with different attributes."""
        doc1 = Document(content="Test", id="1")
        doc2 = Document(content="Different", id="1")
        doc3 = Document(content="Test", id="2")
        doc4 = Document(content="Test", id="1", metadata={"source": "a"})

        assert doc1 != doc2
        assert doc1 != doc3
        assert doc1 != doc4

    def test_from_dict_missing_content_key(self):
        """Test from_dict raises KeyError when 'content' is missing."""
        with pytest.raises(KeyError, match="'content'"):
            Document.from_dict({"id": "test_doc_13"})

    def test_from_dict_with_invalid_types(self):
        """Test from_dict with invalid data types for its values."""
        with pytest.raises(TypeError, match="content must be a string"):
            Document.from_dict({"content": 123, "id": "test_doc_14"})

        with pytest.raises(TypeError, match="metadata must be a dictionary"):
            Document.from_dict({"content": "Test", "metadata": "not-a-dict"})
