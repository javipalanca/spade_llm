"""Tests for text document loaders."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from spade_llm.rag.document_loaders.text import (
    TextLoader,
    DirectoryLoader
)
from spade_llm.rag.core.document import Document


class TestTextLoader:
    """Test cases for TextLoader."""

    def test_init_and_path_handling(self):
        """Test initialization with both string and Path objects."""
        loader_str = TextLoader("/path/to/file.txt")
        assert isinstance(loader_str.file_path, Path)
        assert loader_str.file_path == Path("/path/to/file.txt")

        path_obj = Path("/path/to/file.txt")
        loader_path = TextLoader(path_obj)
        assert loader_path.file_path == path_obj

    def test_extension_support_methods(self):
        """Test the class methods for checking supported extensions."""
        extensions = TextLoader.get_supported_extensions()
        assert isinstance(extensions, set)
        assert '.txt' in extensions and '.md' in extensions

        assert TextLoader.supports_extension('.txt')
        assert TextLoader.supports_extension('md')  # without dot
        assert not TextLoader.supports_extension('.py')

    def test_supports_extension_case_insensitive(self):
        """Test that extension support is case-insensitive."""
        assert TextLoader.supports_extension('.TXT')
        assert TextLoader.supports_extension('.Md')
        assert TextLoader.supports_extension('RST')

    def test_supported_extensions_set(self):
        """Test the supported extensions set."""
        extensions = TextLoader.get_supported_extensions()
        
        assert '.txt' in extensions
        assert '.md' in extensions
        assert '.rst' in extensions
        assert isinstance(extensions, set)

    @pytest.mark.parametrize("method_name", ["load", "load_stream"])
    @pytest.mark.asyncio
    async def test_successful_load(self, method_name):
        """Test successful file loading for both load() and load_stream()."""
        content = "Hello, world!"
        # Use a real temporary file to test the full I/O integration.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            load_method = getattr(loader, method_name)
            
            # Handle async generator vs direct method call
            if method_name == "load_stream":
                result = load_method()
                documents = [doc async for doc in result]
            else:
                documents = await load_method()

            assert len(documents) == 1
            doc = documents[0]
            assert doc.content == content
            assert doc.metadata["source"] == Path(temp_path)
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_with_custom_metadata(self):
        """Test that custom metadata is correctly merged."""
        custom_meta = {"author": "test", "version": 1}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path, metadata=custom_meta)
            documents = await loader.load()
            
            assert len(documents) == 1
            metadata = documents[0].metadata
            assert metadata["author"] == "test"
            assert metadata["version"] == 1
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_custom_metadata_merged(self):
        """Test that custom metadata is merged with default metadata."""
        custom_meta = {"author": "test", "version": 1}
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        ) as f:
            f.write("content")
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path, metadata=custom_meta)
            documents = await loader.load()
            
            assert documents[0].metadata["author"] == "test"
            assert documents[0].metadata["version"] == 1
            assert "source" in documents[0].metadata
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_file_not_found(self):
        """Test that FileNotFoundError is raised for a non-existent file."""
        loader = TextLoader("/nonexistent/file/path.txt")
        with pytest.raises(FileNotFoundError):
            await loader.load()

    @pytest.mark.asyncio
    async def test_load_raises_on_permission_error(self):
        """Test that other IO errors (like PermissionError) are propagated."""
        # Mocking because simulating a permission error is hard.
        with patch('spade_llm.rag.document_loaders.text.aiofiles.open') as mock_open:
            mock_open.side_effect = PermissionError("Access denied")
            loader = TextLoader("/some/protected/file.txt")
            with pytest.raises(PermissionError):
                await loader.load()

    @pytest.mark.asyncio
    async def test_load_unicode_content(self):
        """Test loading file with unicode content."""
        content = "Hello 世界! Привет! مرحبا!"
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', encoding='utf-8', delete=False
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == content
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_different_encoding(self):
        """Test loading file with different encoding."""
        content = "Test content"
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', encoding='latin-1', delete=False
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path, encoding='latin-1')
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == content
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_empty_file(self):
        """Test loading an empty file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        ) as f:
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == ""
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_large_file(self):
        """Test loading a large file."""
        large_content = "a" * (1024 * 1024)
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        ) as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = await loader.load()
            
            assert len(documents) == 1
            assert len(documents[0].content) == len(large_content)
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_metadata_includes_source(self):
        """Test that metadata includes source."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        ) as f:
            f.write("content")
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = await loader.load()
            
            assert "source" in documents[0].metadata
            assert documents[0].metadata["source"] == Path(temp_path)
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_markdown_file(self):
        """Test loading markdown file."""
        content = "# Header\n\nSome **bold** text"
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == content
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_rst_file(self):
        """Test loading RST file."""
        content = "Header\n======\n\nContent"
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.rst', delete=False
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == content
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_stream_yields_one_document(self):
        """Test that load_stream yields exactly one document."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        ) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path)
            documents = []
            async for doc in loader.load_stream():
                documents.append(doc)
            
            assert len(documents) == 1
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_text_loader_invalid_encoding(self):
        """Test TextLoader with invalid encoding."""
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix='.txt', delete=False
        ) as f:
            # Write some binary data
            f.write(b'\x80\x81\x82')
            temp_path = f.name
        
        try:
            loader = TextLoader(temp_path, encoding='utf-8')
            
            # Should raise encoding error
            with pytest.raises(UnicodeDecodeError):
                await loader.load()
        finally:
            Path(temp_path).unlink()


class TestDirectoryLoader:
    """Test cases for DirectoryLoader."""

    def test_init_and_suffix_handling(self):
        """Test initialization and custom suffix logic."""
        loader_default = DirectoryLoader("/path")
        assert '.txt' in loader_default.suffixes

        loader_custom = DirectoryLoader("/path", suffixes=['.py', 'rs'])
        assert '.py' in loader_custom.suffixes
        assert '.rs' in loader_custom.suffixes
        assert '.txt' not in loader_custom.suffixes

    def test_get_loader_for_file(self):
        """Test the internal logic for selecting a loader for a file."""
        loader = DirectoryLoader("/path")
        
        # Test supported file
        txt_loader = loader._get_loader_for_file(Path("a.txt"))
        assert isinstance(txt_loader, TextLoader)
        
        # Test unsupported file
        py_loader = loader._get_loader_for_file(Path("a.py"))
        assert py_loader is None

    def test_get_loader_for_file_parameter_passing(self):
        """Test that _get_loader_for_file passes only accepted parameters to sub-loaders."""
        from unittest.mock import Mock
        
        # Mock loader class with different __init__ signature
        class MockLoader:
            def __init__(self, file_path, metadata=None):
                # Only accepts file_path and metadata, not encoding
                self.file_path = file_path
                self.metadata = metadata or {}
                self.encoding = None  # Should not be set
        
        # Create a DirectoryLoader with the mock loader in its map
        loader = DirectoryLoader(
            "/path", 
            encoding="utf-8",
            metadata={"base": "meta"},
            loader_map={'.mock': MockLoader}
        )
        
        # Test that only accepted parameters are passed
        mock_loader = loader._get_loader_for_file(Path("test.mock"))
        assert isinstance(mock_loader, MockLoader)
        assert mock_loader.file_path == Path("test.mock")
        assert mock_loader.metadata == {"base": "meta"}
        assert mock_loader.encoding is None  # encoding should not be passed

    @pytest.mark.asyncio
    async def test_get_loader_for_file_returns_none_for_unsupported(self):
        """Test that _get_loader_for_file returns None for unsupported files."""
        loader = DirectoryLoader("/tmp")
        
        result = loader._get_loader_for_file(Path("test.py"))
        assert result is None

    @pytest.mark.asyncio
    async def test_load_non_existent_directory(self):
        """Test loading from a directory that does not exist."""
        loader = DirectoryLoader("/non/existent/dir")
        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            await loader.load()

    @pytest.mark.asyncio
    async def test_path_as_string_or_path_object(self):
        """Test that DirectoryLoader accepts both string and Path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # String path
            loader1 = DirectoryLoader(temp_dir)
            assert isinstance(loader1.path, Path)
            
            # Path object
            loader2 = DirectoryLoader(Path(temp_dir))
            assert isinstance(loader2.path, Path)

    @pytest.mark.parametrize("method_name", ["load", "load_stream"])
    @pytest.mark.asyncio
    async def test_load_from_directory(self, method_name):
        """Test successfully loading/streaming from a directory with mixed files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            (dir_path / "doc1.txt").write_text("Text file")
            (dir_path / "doc2.md").write_text("Markdown file")
            (dir_path / "script.py").write_text("Python file") # Should be ignored

            loader = DirectoryLoader(dir_path)
            load_method = getattr(loader, method_name)
            
            if method_name == "load_stream":
                result = load_method()
                documents = [doc async for doc in result]
            else:
                documents = await load_method()
            
            assert len(documents) == 2
            contents = sorted([doc.content for doc in documents])
            assert contents == ["Markdown file", "Text file"]

    @pytest.mark.asyncio
    async def test_load_stream_continues_after_file_error(self):
        """Test that streaming continues if one file fails to load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            (dir_path / "good.txt").write_text("Good content")
            # This file is real, but we will mock its loader to fail
            (dir_path / "bad.md").write_text("This will fail")

            loader = DirectoryLoader(dir_path)
            
            original_get_loader = loader._get_loader_for_file

            def get_loader_with_failure(file_path: Path):
                # Get the real loader instance
                sub_loader = original_get_loader(file_path)
                if sub_loader and "bad.md" in str(file_path):
                    # Create an async generator that raises an exception
                    async def failing_load_stream():
                        raise IOError("Corrupt file")
                        yield  # This will never be reached but makes it a generator
                    
                    sub_loader.load_stream = failing_load_stream
                return sub_loader

            # Use patch to override the instance's method for this test
            with patch.object(loader, '_get_loader_for_file', side_effect=get_loader_with_failure):
                documents = [doc async for doc in loader.load_stream()]
                
                # Loader should skip the bad file and load the good one
                assert len(documents) == 1
                assert documents[0].content == "Good content"

    @pytest.mark.asyncio
    async def test_load_nested_directories(self):
        """Test loading from nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            # Create nested structure
            (dir_path / "subdir1").mkdir()
            (dir_path / "subdir1" / "subdir2").mkdir()
            
            (dir_path / "file1.txt").write_text("Content 1")
            (dir_path / "subdir1" / "file2.txt").write_text("Content 2")
            (dir_path / "subdir1" / "subdir2" / "file3.txt").write_text("Content 3")
            
            loader = DirectoryLoader(dir_path, recursive=True)
            documents = await loader.load()
            
            assert len(documents) == 3
            contents = {doc.content for doc in documents}
            assert "Content 1" in contents
            assert "Content 2" in contents
            assert "Content 3" in contents

    @pytest.mark.asyncio
    async def test_load_non_recursive(self):
        """Test loading without recursion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            # Create nested structure
            (dir_path / "subdir").mkdir()
            (dir_path / "file1.txt").write_text("Root file")
            (dir_path / "subdir" / "file2.txt").write_text("Subdir file")
            
            loader = DirectoryLoader(dir_path, recursive=False, glob_pattern="*.txt")
            documents = await loader.load()
            
            # Should only load root level files
            assert len(documents) == 1
            assert documents[0].content == "Root file"

    @pytest.mark.asyncio
    async def test_custom_glob_pattern(self):
        """Test loading with custom glob pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "file1.txt").write_text("TXT file")
            (dir_path / "file2.md").write_text("MD file")
            (dir_path / "file3.py").write_text("PY file")
            
            # Only load .md files
            loader = DirectoryLoader(dir_path, glob_pattern="*.md")
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == "MD file"

    @pytest.mark.asyncio
    async def test_custom_suffixes_filter(self):
        """Test loading with custom suffix filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "file1.txt").write_text("TXT file")
            (dir_path / "file2.md").write_text("MD file")
            (dir_path / "file3.rst").write_text("RST file")
            
            # Only load .md files using suffixes
            loader = DirectoryLoader(dir_path, suffixes=['.md'])
            documents = await loader.load()
            
            assert len(documents) == 1
            assert documents[0].content == "MD file"

    @pytest.mark.asyncio
    async def test_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DirectoryLoader(temp_dir)
            documents = await loader.load()
            
            assert documents == []

    @pytest.mark.asyncio
    async def test_directory_with_only_unsupported_files(self):
        """Test loading directory with only unsupported files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "file1.py").write_text("Python file")
            (dir_path / "file2.js").write_text("JavaScript file")
            
            loader = DirectoryLoader(dir_path)
            documents = await loader.load()
            
            # Should skip unsupported files
            assert documents == []

    @pytest.mark.asyncio
    async def test_load_stream_multiple_files(self):
        """Test load_stream with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "file1.txt").write_text("Content 1")
            (dir_path / "file2.txt").write_text("Content 2")
            
            loader = DirectoryLoader(dir_path)
            documents = []
            async for doc in loader.load_stream():
                documents.append(doc)
            
            assert len(documents) == 2

    @pytest.mark.asyncio
    async def test_base_metadata_applied_to_all(self):
        """Test that base metadata is applied to all loaded documents."""
        base_metadata = {"source_dir": "test_dir", "batch": 1}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "file1.txt").write_text("Content 1")
            (dir_path / "file2.txt").write_text("Content 2")
            
            loader = DirectoryLoader(dir_path, metadata=base_metadata)
            documents = await loader.load()
            
            assert len(documents) == 2
            for doc in documents:
                assert doc.metadata["source_dir"] == "test_dir"
                assert doc.metadata["batch"] == 1

    @pytest.mark.asyncio
    async def test_mixed_file_types(self):
        """Test loading directory with mixed supported file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "file1.txt").write_text("TXT content")
            (dir_path / "file2.md").write_text("MD content")
            (dir_path / "file3.rst").write_text("RST content")
            (dir_path / "file4.py").write_text("PY content")  # Unsupported
            
            loader = DirectoryLoader(dir_path)
            documents = await loader.load()
            
            # Should load 3 supported files
            assert len(documents) == 3
            contents = {doc.content for doc in documents}
            assert "TXT content" in contents
            assert "MD content" in contents
            assert "RST content" in contents

    @pytest.mark.asyncio
    async def test_symlink_handling(self):
        """Test handling of symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            # Create a file and a symlink to it
            real_file = dir_path / "real.txt"
            real_file.write_text("Real content")
            
            # Note: symlink creation may fail on Windows without admin rights
            try:
                symlink = dir_path / "link.txt"
                symlink.symlink_to(real_file)
                
                loader = DirectoryLoader(dir_path)
                documents = await loader.load()
                
                # Behavior may vary - could load both or just one
                assert len(documents) >= 1
            except OSError:
                # Skip test if symlinks not supported
                pytest.skip("Symlinks not supported on this system")

    @pytest.mark.asyncio
    async def test_hidden_files_handling(self):
        """Test handling of hidden files (starting with .)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            (dir_path / "visible.txt").write_text("Visible")
            (dir_path / ".hidden.txt").write_text("Hidden")
            
            loader = DirectoryLoader(dir_path)
            documents = await loader.load()
            
            # The default glob should find both files.
            assert len(documents) == 2
            contents = sorted([doc.content for doc in documents])
            assert contents == ["Hidden", "Visible"]

