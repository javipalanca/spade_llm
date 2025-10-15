"""Tests for character text splitters."""

import pytest
from spade_llm.rag import (
    Document,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class TestCharacterTextSplitter:
    """Test cases for CharacterTextSplitter."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        splitter = CharacterTextSplitter()
        assert splitter.chunk_size == 2000
        assert splitter.chunk_overlap == 200
        assert splitter.separator == "\n\n"
        assert splitter.is_separator_regex is False

    def test_initialization_custom(self):
        """Test custom initialization."""
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n",
        )
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
        assert splitter.separator == "\n"

    def test_split_text_simple(self):
        """Test splitting simple text with precise size and overlap checks."""
        splitter = CharacterTextSplitter(
            chunk_size=20,
            chunk_overlap=5,
            separator=" "
        )
        text = "This is a simple test text that should be split"
        chunks = splitter.split_text(text)
        expected_chunks = ['This is a simple', 'test text that', 'that should be split']

        assert chunks == expected_chunks
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_split_text_empty(self):
        """Test splitting empty text."""
        splitter = CharacterTextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_text_invalid_input(self):
        """Test splitting non-string input."""
        splitter = CharacterTextSplitter()
        result = splitter.split_text(None) # type: ignore
        assert result == []

    def test_split_documents(self):
        """Test splitting documents."""
        splitter = CharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            separator="\n\n"
        )
        
        documents = [
            Document(
                id="doc1",
                content="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
                metadata={"source": "test1"}
            ),
            Document(
                id="doc2",
                content="Another document.\n\nWith more content.",
                metadata={"source": "test2"}
            )
        ]
        
        chunks = splitter.split_documents(documents)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Verify metadata is preserved and chunks have IDs
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.id  # Chunk should have an ID
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] in ["test1", "test2"]

    def test_split_text_with_separator_regex(self):
        """Test splitting with regex separator."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separator=r"\s+",
            is_separator_regex=True
        )
        
        text = "Word1   Word2\nWord3\tWord4"
        chunks = splitter.split_text(text)
        expected_chunks = ['Word1', 'Word2', 'Word3', 'Word4']
        
        assert chunks == expected_chunks

    def test_split_with_overlap(self):
        """Test that chunk overlap works correctly with explicit verification."""
        splitter = CharacterTextSplitter(
            chunk_size=20,
            chunk_overlap=5,
            separator=" "
        )
        
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = splitter.split_text(text)
        expected_chunks = ['word1 word2 word3', 'word3 word4 word5', 'word5 word6 word7', 'word7 word8']
        
        assert chunks == expected_chunks

    def test_split_text_regex_separator(self):
        """Test splitting with a regex separator."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separator=r"\d+\.",  # Match numbered lists
            is_separator_regex=True
        )
        
        text = "Item 1. First item 2. Second item ... 11. Eleventh item"
        chunks = splitter.split_text(text)
        expected_chunks = ['Item', 'First item', 'Second item ...', 'Eleventh item']

        assert chunks == expected_chunks

    def test_keep_separator_start(self):
        """Test keeping separator at start of chunks."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separator="<SEP>",
            keep_separator="start"
        )
        
        text = "Line 1<SEP>Line 2<SEP>Line 3"
        chunks = splitter.split_text(text)
        expected_chunks = ['Line 1', '<SEP>Line 2', '<SEP>Line 3']

        assert chunks == expected_chunks

    def test_keep_separator_end(self):
        """Test keeping separator at end of chunks."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separator="<SEP>",
            keep_separator="end"
        )
        
        text = "Line 1<SEP>Line 2<SEP>Line 3"
        chunks = splitter.split_text(text)
        expected_chunks = ['Line 1<SEP>', 'Line 2<SEP>', 'Line 3']

        assert chunks == expected_chunks


    def test_keep_separator_with_regex(self):
        """Test keeping separator at start with regex separator."""
        splitter = CharacterTextSplitter(
            chunk_size=6,
            chunk_overlap=0,
            separator=r"\n+",  # One or more newlines
            is_separator_regex=True,
            keep_separator="start"
        )
        
        text = "One\n\nTwo\nThree"
        chunks = splitter.split_text(text)
        
        # Note: With regex separators, keep_separator behavior may not preserve separators
        # as they are consumed by the regex match
        expected_chunks = ['One', 'Two', 'Three']

        assert chunks == expected_chunks

    def test_empty_separator_with_regex(self):
        """Test empty string as regex separator doesn't cause infinite loops."""
        splitter = CharacterTextSplitter(
            chunk_size=3,
            chunk_overlap=0,
            separator="",
            is_separator_regex=True
        )
        
        text = "HelloWorld"
        chunks = splitter.split_text(text)
        expected_chunks = ['Hel', 'loW', 'orl', 'd']

        assert chunks == expected_chunks

    def test_strip_whitespace_enabled(self):
        """Test that whitespace is stripped when enabled."""
        splitter = CharacterTextSplitter(
            chunk_size=15,
            chunk_overlap=0,
            separator="\n\n",
            strip_whitespace=True
        )
        
        text = "  Paragraph one  \n\n  Paragraph two  \n\n  Para three  "
        chunks = splitter.split_text(text)
        expected_chunks = ['Paragraph one', 'Paragraph two', 'Para three']

        assert chunks == expected_chunks

    def test_strip_whitespace_disabled(self):
        """Test that whitespace is preserved when disabled."""
        splitter = CharacterTextSplitter(
            chunk_size=15,
            chunk_overlap=0,
            separator="\n",
            strip_whitespace=False
        )
        
        text = "  Line one  \n  Line two  \n  Line three  "
        chunks = splitter.split_text(text)
        expected_chunks = ['  Line one  ', '  Line two  ', '  Line three  ']

        assert chunks == expected_chunks

    def test_custom_length_function(self):
        """Test using a custom length function."""
        # Count words instead of characters
        def word_count(text):
            return len(text.split())
        
        splitter = CharacterTextSplitter(
            chunk_size=5,  # 5 words
            chunk_overlap=1,  # 1 word overlap
            separator=" ",
            length_function=word_count
        )
        
        text = "one two three four five six seven eight nine ten"
        chunks = splitter.split_text(text)
        expected_chunks = ['one two three four five', 'five six seven eight nine', 'nine ten']

        assert chunks == expected_chunks

    def test_split_very_long_text(self):
        """Test splitting very long text."""
        splitter = CharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separator=" "
        )
        
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 99
        # Check that the chunks are of the expected length
        assert len(chunks[0]) == 94
        assert len(chunks[-1]) == 71

    def test_single_word_longer_than_chunk_size(self):
        """Test handling when a single word is longer than chunk_size."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separator=" "
        )
        
        text = "short verylongword short"
        chunks = splitter.split_text(text)
        expected_chunks = ['short', 'verylongword', 'short']

        assert chunks == expected_chunks

    def test_empty_separator(self):
        """Test splitting with empty string separator."""
        splitter = CharacterTextSplitter(
            chunk_size=2,
            chunk_overlap=1,
            separator=""
        )
        
        text = "Hello"
        chunks = splitter.split_text(text)
        expected_chunks = ['He', 'el', 'll', 'lo']
        
        assert chunks == expected_chunks

    def test_unicode_text_splitting(self):
        """Test splitting text with unicode characters."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=5,
            separator=" "
        )
        
        text = "Hello 世界 Привет مرحبا שלום नमस्ते"
        chunks = splitter.split_text(text)
        expected_chunks = ['Hello 世界', '世界 Привет', 'مرحبا שלום', 'नमस्ते']

        assert chunks == expected_chunks

    def test_only_separators_in_text(self):
        """Test text that is only separators."""
        splitter = CharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separator="\n"
        )
        
        text = "\n\n\n\n"
        chunks = splitter.split_text(text)
        
        # Return empty list since there's no actual content
        assert chunks == []

    def test_lookahead_regex_separator(self):
        """Test splitting with lookahead regex pattern."""
        splitter = CharacterTextSplitter(
            chunk_size=15,
            chunk_overlap=0,
            separator=r"(?=\n)",  # Lookahead for newline
            is_separator_regex=True,
            keep_separator=False
        )
        
        text = "First line.\nSecond line.\nThird line."
        chunks = splitter.split_text(text)
        expected_chunks = ['First line.', 'Second line.', 'Third line.']

        assert chunks == expected_chunks

    def test_chunk_overlap_larger_validation(self):
        """Test that error is raised when overlap > chunk_size."""
        with pytest.raises(ValueError, match="larger chunk overlap.*should be smaller"):
            CharacterTextSplitter(chunk_size=100, chunk_overlap=150)

    def test_zero_chunk_size_validation(self):
        """Test that error is raised for zero chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            CharacterTextSplitter(chunk_size=0)

    def test_negative_chunk_overlap_validation(self):
        """Test that error is raised for negative chunk_overlap."""
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            CharacterTextSplitter(chunk_overlap=-10)

    
class TestRecursiveCharacterTextSplitter:
    """Test cases for RecursiveCharacterTextSplitter."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        splitter = RecursiveCharacterTextSplitter()
        assert splitter.chunk_size == 2000
        assert splitter.chunk_overlap == 200
        assert len(splitter.separators) > 0

    def test_initialization_custom_separators(self):
        """Test custom separators."""
        custom_seps = ["\n\n", "\n", " "]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=custom_seps
        )
        assert splitter.separators == custom_seps

    def test_hierarchical_splitting_verifies_recursion(self):
        """Test that recursive splitting actually uses multiple separator levels."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=5,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Text where first separator creates large chunks
        # that need further splitting by secondary separators
        text = """First paragraph has many words here.

Second paragraph also has many words that exceed the chunk size."""
        
        chunks = splitter.split_text(text)
        expected_chunks = ['First paragraph has many words', 'here.', 'Second paragraph also has', 'has many words that exceed', 'the chunk size.']

        assert chunks == expected_chunks

    def test_split_documents_preserves_metadata(self):
        """Test that document splitting preserves metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=5
        )
        
        doc = Document(
            id="test_doc",
            content="This is a test document with multiple sentences. It should be split into chunks.",
            metadata={"year": 2025, "author": "tester"}
        )
        
        chunks = splitter.split_documents([doc])
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk.id, str)  # Chunk should have a UUID
            assert len(chunk.id) > 0
            assert chunk.metadata["year"] == 2025
            assert chunk.metadata["author"] == "tester"
            assert "chunk_index" in chunk.metadata

    def test_split_text_empty(self):
        """Test splitting empty text."""
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_text_shorter_than_chunk_size(self):
        """Test text shorter than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        text = "Short text"
        chunks = splitter.split_text(text)
        expected_chunks = ['Short text']

        assert chunks == expected_chunks

    def test_create_documents(self):
        """Test creating documents from texts."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        documents = splitter.create_documents(texts, metadatas)
        
        # create_documents creates documents and then splits them, so we get chunks
        assert len(documents) == 3
        for doc in documents:
            assert isinstance(doc, Document)
            assert doc.id
            assert "id" in doc.metadata

    def test_custom_separators_order(self):
        """Test that separators are tried in order."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=0,
            separators=["###", "##", "#", " "]
        )
        
        text = "### Header\n## Subheader\n# Item\nSome text here"
        chunks = splitter.split_text(text)
        expected_chunks = ['### Header', '## Subheader', '# Item\nSome text here']

        assert chunks == expected_chunks

    def test_no_separator_found(self):
        """Test behavior when no separator is found in text - returns whole text as single chunk."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,
            chunk_overlap=0,
            separators=["|", "^", "~"]  # None of these are in the text
        )
        
        text = "This text has no special separators at all"
        chunks = splitter.split_text(text)
        expected_chunks = ['This text has no special separators at all']

        assert chunks == expected_chunks

    def test_no_separators_falls_back_to_character_split(self):
        """Test that when no separators match, splitter uses character-level split."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=2,
            separators=["\n\n", "\n", " ", ""]  # Empty string for character split
        )
        
        # Text with no spaces or newlines
        text = "ThisIsAVeryLongWordWithoutSpaces"
        chunks = splitter.split_text(text)
        expected_chunks = ['ThisIsAVer', 'eryLongWor', 'ordWithout', 'utSpaces']

        assert chunks == expected_chunks

    def test_keep_separator_recursive(self):
        """Test separator retention in recursive splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=15,
            chunk_overlap=0,
            separators=["\n\n", "\n"],
            keep_separator=True
        )
        
        text = "Paragraph 1\n\nParagraph 2\nLine\n\nParagraph 3"
        chunks = splitter.split_text(text)
        expected_chunks = ['Paragraph 1', 'Paragraph 2', 'Line', 'Paragraph 3']

        assert chunks == expected_chunks

    def test_keep_separator_with_multiple_separators(self):
        """Test how keep_separator behaves with different separators in recursive splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=21,
            chunk_overlap=0,
            separators=["<SEP><SEP>", "<SEP>"],
            keep_separator="start"
        )

        text =  "Paragraph 1<SEP><SEP>Paragraph 2<SEP>Line<SEP><SEP>Paragraph 3"
        chunks = splitter.split_text(text)
        expected_chunks = ['Paragraph 1', '<SEP><SEP>Paragraph 2', '<SEP>Line', '<SEP><SEP>Paragraph 3']

        assert chunks == expected_chunks

    def test_create_documents_with_chunk_index(self):
        """Test that create_documents adds chunk_index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,
            chunk_overlap=5
        )
        
        texts = ["Short text that will be split into multiple chunks for testing"]
        documents = splitter.create_documents(texts)
        
        # Check chunk indices
        for i, doc in enumerate(documents):
            assert "chunk_index" in doc.metadata
            assert doc.metadata["chunk_index"] == i

    def test_add_start_index_in_metadata(self):
        """Test that start_index is added when enabled and correctly calculated with overlap."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=5,
            add_start_index=True
        )
        
        # Use text with repeating phrases to test start_index calculation
        text = "This is a text that will be split into chunks. This is a text that continues."
        docs = splitter.create_documents([text])
        
        expected_starts = [0, 25, 52]
        expected_contents = ['This is a text that will be', 'be split into chunks. This is', 'is a text that continues.']

        for doc, expected_start, expected_content in zip(docs, expected_starts, expected_contents):
            assert "start_index" in doc.metadata
            assert doc.metadata["start_index"] == expected_start
            assert doc.content == expected_content

    def test_no_start_index_when_disabled(self):
        """Test that start_index is not added when disabled."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=5,
            add_start_index=False
        )
        
        text = "This is a text that will be split into chunks"
        docs = splitter.create_documents([text])
        
        for doc in docs:
            assert "start_index" not in doc.metadata

    def test_metadata_deep_copy(self):
        """Test that metadata is deep copied, not referenced."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=5
        )
        
        original_metadata = {"nested": {"key": "value"}}
        doc = Document(
            content="Long text here for splitting into chunks",
            metadata=original_metadata
        )
        
        chunks = splitter.split_documents([doc])
        
        # Modify chunk metadata
        if len(chunks) > 0:
            chunks[0].metadata["nested"]["key"] = "modified"
            
            # Original should not be affected
            assert original_metadata["nested"]["key"] == "value"

