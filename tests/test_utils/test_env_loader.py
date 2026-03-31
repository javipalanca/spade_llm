"""Tests for environment variable loading utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from spade_llm.utils.env_loader import _get_env_file_variables, _manual_load_env, get_memory_path, load_env_vars


class TestLoadEnvVars:
    """Test the load_env_vars function."""

    def test_load_env_vars_dotenv_not_available(self):
        """Test fallback to manual loading when dotenv is not available."""
        with patch("spade_llm.utils.env_loader._manual_load_env") as mock_manual:
            mock_manual.return_value = {"FALLBACK_VAR": "fallback_value"}

            # Mock ImportError when trying to import dotenv
            with patch("spade_llm.utils.env_loader.logger") as mock_logger:
                with patch("builtins.__import__", side_effect=ImportError("No module named 'dotenv'")):
                    result = load_env_vars(".env")

                    # Verify manual loading was called
                    mock_manual.assert_called_once_with(".env")
                    mock_logger.warning.assert_called_with(
                        "python-dotenv not installed, falling back to manual .env parsing"
                    )
                    assert result == {"FALLBACK_VAR": "fallback_value"}


class TestGetEnvFileVariables:
    """Test the _get_env_file_variables function."""

    def test_get_env_file_variables_success(self):
        """Test extracting variables from .env file."""
        env_content = """# Comment line
TEST_VAR=test_value
ANOTHER_VAR=another_value
QUOTED_VAR="quoted_value"
SINGLE_QUOTED='single_quoted_value'

# Another comment
EMPTY_VAR=
"""

        with patch("builtins.open", mock_open(read_data=env_content)):
            mock_path = Mock()
            result = _get_env_file_variables(mock_path)

            expected = {
                "TEST_VAR": "test_value",
                "ANOTHER_VAR": "another_value",
                "QUOTED_VAR": "quoted_value",
                "SINGLE_QUOTED": "single_quoted_value",
                "EMPTY_VAR": "",
            }

            assert result == expected

    def test_get_env_file_variables_with_quotes(self):
        """Test proper handling of quoted values."""
        env_content = """DOUBLE_QUOTED="double quoted value"
SINGLE_QUOTED='single quoted value'
MIXED_QUOTES="value with 'internal' quotes"
NO_QUOTES=no_quotes_value
"""

        with patch("builtins.open", mock_open(read_data=env_content)):
            mock_path = Mock()
            result = _get_env_file_variables(mock_path)

            expected = {
                "DOUBLE_QUOTED": "double quoted value",
                "SINGLE_QUOTED": "single quoted value",
                "MIXED_QUOTES": "value with 'internal' quotes",
                "NO_QUOTES": "no_quotes_value",
            }

            assert result == expected

    def test_get_env_file_variables_ignores_comments_and_empty_lines(self):
        """Test that comments and empty lines are ignored."""
        env_content = """# This is a comment
VALID_VAR=valid_value

# Another comment
ANOTHER_VAR=another_value

   # Indented comment
"""

        with patch("builtins.open", mock_open(read_data=env_content)):
            mock_path = Mock()
            result = _get_env_file_variables(mock_path)

            expected = {"VALID_VAR": "valid_value", "ANOTHER_VAR": "another_value"}

            assert result == expected

    def test_get_env_file_variables_handles_malformed_line(self):
        """Test handling of malformed lines."""
        env_content = """VALID_VAR=valid_value
MALFORMED_LINE_NO_EQUALS
"""

        with patch("builtins.open", mock_open(read_data=env_content)):
            mock_path = Mock()

            # Should raise ValueError due to malformed line
            with pytest.raises(ValueError):
                _get_env_file_variables(mock_path)

    def test_get_env_file_variables_with_equals_in_value(self):
        """Test handling of equals signs in values."""
        env_content = """URL_VAR=http://example.com/path?param=value
COMPLEX_VAR=key=value&another=test
"""

        with patch("builtins.open", mock_open(read_data=env_content)):
            mock_path = Mock()
            result = _get_env_file_variables(mock_path)

            expected = {"URL_VAR": "http://example.com/path?param=value", "COMPLEX_VAR": "key=value&another=test"}

            assert result == expected

    def test_get_env_file_variables_with_special_characters(self):
        """Test handling of special characters in values."""
        env_content = """SPECIAL_CHARS=!@#$%^&*()_+-=[]{}|;:,.<>?
UNICODE_VAR=тест_значение
"""

        with patch("builtins.open", mock_open(read_data=env_content)):
            mock_path = Mock()
            result = _get_env_file_variables(mock_path)

            expected = {"SPECIAL_CHARS": "!@#$%^&*()_+-=[]{}|;:,.<>?", "UNICODE_VAR": "тест_значение"}

            assert result == expected

    def test_get_env_file_variables_with_file_read_error(self):
        """Test _get_env_file_variables when file read fails."""
        mock_path = Mock()

        # Mock open to raise IOError
        with patch("builtins.open", side_effect=IOError("File read error")):
            with pytest.raises(IOError):
                _get_env_file_variables(mock_path)


class TestGetMemoryPath:
    """Test the get_memory_path function."""

    def test_get_memory_path_default(self):
        """Test getting default memory path."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("spade_llm.utils.env_loader.Path") as mock_path_class:
                with patch("spade_llm.utils.env_loader.logger"):
                    mock_path_instance = Mock()
                    mock_path_instance.mkdir = Mock()
                    mock_path_instance.absolute.return_value = Path("/default/path")
                    mock_path_class.return_value = mock_path_instance

                    result = get_memory_path()

                    # Verify default path was used
                    mock_path_class.assert_called_with("spade_llm/data/agent_memory")
                    mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)
                    assert result == mock_path_instance

    def test_get_memory_path_custom_env_var(self):
        """Test getting custom memory path from environment variable."""
        custom_path = "/custom/memory/path"

        with patch.dict(os.environ, {"SPADE_LLM_MEMORY_PATH": custom_path}):
            with patch("spade_llm.utils.env_loader.Path") as mock_path_class:
                with patch("spade_llm.utils.env_loader.logger"):
                    mock_path_instance = Mock()
                    mock_path_instance.mkdir = Mock()
                    mock_path_instance.absolute.return_value = Path(custom_path)
                    mock_path_class.return_value = mock_path_instance

                    result = get_memory_path()

                    # Verify custom path was used
                    mock_path_class.assert_called_with(custom_path)
                    mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)
                    assert result == mock_path_instance

    def test_get_memory_path_creation_fails_permission_error(self):
        """Test fallback when directory creation fails due to permissions."""
        custom_path = "/restricted/memory/path"

        with patch.dict(os.environ, {"SPADE_LLM_MEMORY_PATH": custom_path}):
            with patch("spade_llm.utils.env_loader.Path") as mock_path_class:
                with patch("spade_llm.utils.env_loader.logger") as mock_logger:
                    # First call (custom path) - raises PermissionError
                    custom_path_instance = Mock()
                    custom_path_instance.mkdir.side_effect = PermissionError("Permission denied")

                    # Second call (fallback path) - succeeds
                    fallback_path_instance = Mock()
                    fallback_path_instance.mkdir = Mock()
                    fallback_path_instance.absolute.return_value = Path("/fallback/path")

                    mock_path_class.side_effect = [custom_path_instance, fallback_path_instance]

                    result = get_memory_path()

                    # Verify fallback was used
                    assert mock_path_class.call_count == 2
                    mock_path_class.assert_any_call(custom_path)
                    mock_path_class.assert_any_call("spade_llm/data/agent_memory")

                    # Verify fallback directory was created
                    fallback_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

                    # Verify logging
                    mock_logger.error.assert_called()
                    mock_logger.info.assert_called()

                    assert result == fallback_path_instance

    def test_get_memory_path_creation_fails_os_error(self):
        """Test fallback when directory creation fails due to OS error."""
        custom_path = "/invalid/memory/path"

        with patch.dict(os.environ, {"SPADE_LLM_MEMORY_PATH": custom_path}):
            with patch("spade_llm.utils.env_loader.Path") as mock_path_class:
                with patch("spade_llm.utils.env_loader.logger") as mock_logger:
                    # First call (custom path) - raises OSError
                    custom_path_instance = Mock()
                    custom_path_instance.mkdir.side_effect = OSError("Invalid path")

                    # Second call (fallback path) - succeeds
                    fallback_path_instance = Mock()
                    fallback_path_instance.mkdir = Mock()
                    fallback_path_instance.absolute.return_value = Path("/fallback/path")

                    mock_path_class.side_effect = [custom_path_instance, fallback_path_instance]

                    result = get_memory_path()

                    # Verify fallback was used
                    assert mock_path_class.call_count == 2
                    mock_logger.error.assert_called()
                    mock_logger.info.assert_called()
                    assert result == fallback_path_instance

    def test_get_memory_path_logs_debug_info(self):
        """Test that debug information is logged."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("spade_llm.utils.env_loader.Path") as mock_path_class:
                with patch("spade_llm.utils.env_loader.logger") as mock_logger:
                    mock_path_instance = Mock()
                    mock_path_instance.mkdir = Mock()
                    # Use a Windows-style path for consistency
                    mock_path_instance.absolute.return_value = Path("C:\\debug\\path")
                    mock_path_class.return_value = mock_path_instance

                    result = get_memory_path()

                    # Check for Windows-style path in debug log
                    mock_logger.debug.assert_called_with("Memory path configured: C:\\debug\\path")
                    assert result == mock_path_instance


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_get_memory_path_integration_with_real_paths(self):
        """Test get_memory_path with real path operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_memory_path = os.path.join(temp_dir, "custom_memory")

            with patch.dict(os.environ, {"SPADE_LLM_MEMORY_PATH": custom_memory_path}):
                with patch("spade_llm.utils.env_loader.logger"):
                    result = get_memory_path()

                    # Verify path was created and returned
                    assert result == Path(custom_memory_path)
                    assert result.exists()
                    assert result.is_dir()

    def test_module_imports_correctly(self):
        """Test that all functions can be imported without errors."""
        from spade_llm.utils.env_loader import _get_env_file_variables, get_memory_path, load_env_vars

        # Verify all functions are callable
        assert callable(load_env_vars)
        assert callable(_manual_load_env)
        assert callable(_get_env_file_variables)
        assert callable(get_memory_path)

    def test_logger_usage_in_load_env_vars(self):
        """Test that logger is used in load_env_vars."""
        with patch("spade_llm.utils.env_loader.logger") as mock_logger:
            with patch("spade_llm.utils.env_loader._manual_load_env") as mock_manual:
                mock_manual.return_value = {}

                with patch("builtins.__import__", side_effect=ImportError("No module named 'dotenv'")):
                    load_env_vars(".env")

                    # Verify warning was logged
                    mock_logger.warning.assert_called_with(
                        "python-dotenv not installed, falling back to manual .env parsing"
                    )

    def test_logger_usage_in_get_memory_path(self):
        """Test that logger is used in get_memory_path."""
        with patch("spade_llm.utils.env_loader.logger") as mock_logger:
            with patch("spade_llm.utils.env_loader.Path") as mock_path_class:
                mock_path_instance = Mock()
                mock_path_instance.mkdir = Mock()
                mock_path_instance.absolute.return_value = Path("C:\\test\\path")
                mock_path_class.return_value = mock_path_instance

                with patch.dict(os.environ, {}, clear=True):
                    get_memory_path()

                    # Verify debug was logged
                    mock_logger.debug.assert_called_with("Memory path configured: C:\\test\\path")
