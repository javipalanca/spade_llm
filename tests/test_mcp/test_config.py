"""Tests for MCP configuration classes."""

from pathlib import Path

import pytest

from spade_llm.mcp.config import MCPServerConfig, SseServerConfig, StdioServerConfig


class TestMCPServerConfig:
    """Test MCPServerConfig base class."""

    def test_abstract_methods(self):
        """Test that subclass must implement abstract methods."""

        # Create a concrete subclass for testing
        class ConcreteMCPConfig(MCPServerConfig):
            pass

        # Should be able to instantiate concrete subclass
        config = ConcreteMCPConfig("test_server", cache_tools=False)
        assert config.name == "test_server"
        assert config.cache_tools is False


class TestStdioServerConfig:
    """Test StdioServerConfig class."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        config = StdioServerConfig(name="test_server", command="python")

        assert config.name == "test_server"
        assert config.command == "python"
        assert config.args == []
        assert config.env is None
        assert config.cwd is None
        assert config.encoding == "utf-8"
        assert config.encoding_error_handler == "strict"
        assert config.read_timeout_seconds == 5.0
        assert config.cache_tools is False

    def test_init_full_parameters(self):
        """Test initialization with all parameters."""
        env_vars = {"PATH": "/usr/bin", "PYTHONPATH": "/opt/lib"}
        cwd_path = Path("/tmp/workspace")

        config = StdioServerConfig(
            name="full_server",
            command="node",
            args=["server.js", "--port", "8080"],
            env=env_vars,
            cwd=cwd_path,
            encoding="utf-16",
            encoding_error_handler="ignore",
            read_timeout_seconds=10.5,
            cache_tools=True,
        )

        assert config.name == "full_server"
        assert config.command == "node"
        assert config.args == ["server.js", "--port", "8080"]
        assert config.env == env_vars
        assert config.cwd == cwd_path
        assert config.encoding == "utf-16"
        assert config.encoding_error_handler == "ignore"
        assert config.read_timeout_seconds == 10.5
        assert config.cache_tools is True

    def test_init_with_string_cwd(self):
        """Test initialization with string cwd."""
        config = StdioServerConfig(name="test_server", command="python", cwd="/home/user/project")

        assert config.cwd == "/home/user/project"
        assert isinstance(config.cwd, str)

    def test_init_with_path_cwd(self):
        """Test initialization with Path cwd."""
        cwd_path = Path("/home/user/project")

        config = StdioServerConfig(name="test_server", command="python", cwd=cwd_path)

        assert config.cwd == cwd_path
        assert isinstance(config.cwd, Path)

    def test_post_init_validation_success(self):
        """Test __post_init__ validation with valid command."""
        # Should not raise exception
        config = StdioServerConfig(name="test_server", command="python")

        assert config.command == "python"

    def test_post_init_validation_none_command(self):
        """Test __post_init__ validation with None command."""
        with pytest.raises(ValueError, match="command is required for StdioServerConfig"):
            StdioServerConfig(name="test_server", command=None)

    def test_post_init_validation_with_field_default(self):
        """Test __post_init__ when command defaults to None."""
        with pytest.raises(ValueError, match="command is required for StdioServerConfig"):
            # Using field defaults (command defaults to None)
            StdioServerConfig(name="test_server")

    def test_default_args_list(self):
        """Test that args defaults to empty list."""
        config = StdioServerConfig(name="test_server", command="python")

        assert config.args == []
        assert isinstance(config.args, list)

        # Should be able to modify
        config.args.append("--version")
        assert config.args == ["--version"]

    def test_empty_args_list(self):
        """Test with explicitly empty args list."""
        config = StdioServerConfig(name="test_server", command="python", args=[])

        assert config.args == []

    def test_environment_variables(self):
        """Test environment variables handling."""
        env_vars = {"DEBUG": "true", "API_KEY": "secret-key-123", "PORT": "3000", "PATH": "/usr/local/bin:/usr/bin"}

        config = StdioServerConfig(name="test_server", command="python", env=env_vars)

        assert config.env == env_vars
        assert config.env["DEBUG"] == "true"
        assert config.env["API_KEY"] == "secret-key-123"

    def test_encoding_options(self):
        """Test different encoding options."""
        # UTF-8 (default)
        config1 = StdioServerConfig(name="test", command="python")
        assert config1.encoding == "utf-8"
        assert config1.encoding_error_handler == "strict"

        # UTF-16 with ignore errors
        config2 = StdioServerConfig(name="test", command="python", encoding="utf-16", encoding_error_handler="ignore")
        assert config2.encoding == "utf-16"
        assert config2.encoding_error_handler == "ignore"

        # ASCII with replace errors
        config3 = StdioServerConfig(name="test", command="python", encoding="ascii", encoding_error_handler="replace")
        assert config3.encoding == "ascii"
        assert config3.encoding_error_handler == "replace"

    def test_timeout_values(self):
        """Test different timeout values."""
        # Default timeout
        config1 = StdioServerConfig(name="test", command="python")
        assert config1.read_timeout_seconds == 5.0

        # Custom timeout
        config2 = StdioServerConfig(name="test", command="python", read_timeout_seconds=30.5)
        assert config2.read_timeout_seconds == 30.5

        # Zero timeout
        config3 = StdioServerConfig(name="test", command="python", read_timeout_seconds=0.0)
        assert config3.read_timeout_seconds == 0.0


class TestSseServerConfig:
    """Test SseServerConfig class."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        config = SseServerConfig(name="sse_server", url="https://api.example.com/sse")

        assert config.name == "sse_server"
        assert config.url == "https://api.example.com/sse"
        assert config.headers is None
        assert config.timeout == 5.0
        assert config.sse_read_timeout == 300.0
        assert config.cache_tools is False

    def test_init_full_parameters(self):
        """Test initialization with all parameters."""
        headers = {"Authorization": "Bearer token123", "Content-Type": "application/json", "X-API-Key": "secret-key"}

        config = SseServerConfig(
            name="full_sse_server",
            url="https://api.example.com/sse/stream",
            headers=headers,
            timeout=15.0,
            sse_read_timeout=600.0,
            cache_tools=True,
        )

        assert config.name == "full_sse_server"
        assert config.url == "https://api.example.com/sse/stream"
        assert config.headers == headers
        assert config.timeout == 15.0
        assert config.sse_read_timeout == 600.0
        assert config.cache_tools is True

    def test_post_init_validation_success(self):
        """Test __post_init__ validation with valid URL."""
        # Should not raise exception
        config = SseServerConfig(name="test_server", url="https://example.com")

        assert config.url == "https://example.com"

    def test_post_init_validation_none_url(self):
        """Test __post_init__ validation with None URL."""
        with pytest.raises(ValueError, match="url is required for SseServerConfig"):
            SseServerConfig(name="test_server", url=None)

    def test_post_init_validation_with_field_default(self):
        """Test __post_init__ when URL defaults to None."""
        with pytest.raises(ValueError, match="url is required for SseServerConfig"):
            # Using field defaults (url defaults to None)
            SseServerConfig(name="test_server")

    def test_http_headers(self):
        """Test HTTP headers handling."""
        headers = {"User-Agent": "SPADE-LLM/1.0", "Accept": "text/event-stream", "Cache-Control": "no-cache"}

        config = SseServerConfig(name="test_server", url="https://api.example.com", headers=headers)

        assert config.headers == headers
        assert config.headers["User-Agent"] == "SPADE-LLM/1.0"
        assert config.headers["Accept"] == "text/event-stream"

    def test_empty_headers(self):
        """Test with empty headers dictionary."""
        config = SseServerConfig(name="test_server", url="https://api.example.com", headers={})

        assert config.headers == {}

    def test_timeout_values(self):
        """Test different timeout values."""
        # Default timeouts
        config1 = SseServerConfig(name="test", url="https://example.com")
        assert config1.timeout == 5.0
        assert config1.sse_read_timeout == 300.0

        # Custom timeouts
        config2 = SseServerConfig(name="test", url="https://example.com", timeout=30.0, sse_read_timeout=900.0)
        assert config2.timeout == 30.0
        assert config2.sse_read_timeout == 900.0

        # Very short timeouts
        config3 = SseServerConfig(name="test", url="https://example.com", timeout=0.1, sse_read_timeout=1.0)
        assert config3.timeout == 0.1
        assert config3.sse_read_timeout == 1.0

    def test_url_formats(self):
        """Test different URL formats."""
        urls = [
            "https://api.example.com/sse",
            "http://localhost:8080/events",
            "https://subdomain.example.org:9443/stream/events",
            "https://api.example.com/v1/sse?token=abc123",
        ]

        for url in urls:
            config = SseServerConfig(name="test", url=url)
            assert config.url == url


class TestMCPConfigEdgeCases:
    """Test edge cases for MCP configuration classes."""

    def test_stdio_config_with_empty_string_command(self):
        """Test StdioServerConfig with empty string command."""
        # Empty string should be valid (though might not work in practice)
        config = StdioServerConfig(name="test_server", command="")

        assert config.command == ""

    def test_stdio_config_with_whitespace_command(self):
        """Test StdioServerConfig with whitespace-only command."""
        config = StdioServerConfig(name="test_server", command="   ")

        assert config.command == "   "

    def test_sse_config_with_empty_string_url(self):
        """Test SseServerConfig with empty string URL."""
        # Empty string should be valid (though might not work in practice)
        config = SseServerConfig(name="test_server", url="")

        assert config.url == ""

    def test_very_long_names(self):
        """Test with very long server names."""
        long_name = "a" * 1000

        stdio_config = StdioServerConfig(name=long_name, command="python")
        assert stdio_config.name == long_name

        sse_config = SseServerConfig(name=long_name, url="https://example.com")
        assert sse_config.name == long_name

    def test_special_characters_in_names(self):
        """Test with special characters in server names."""
        special_name = "server-name_with.special@chars#$%"

        stdio_config = StdioServerConfig(name=special_name, command="python")
        assert stdio_config.name == special_name

        sse_config = SseServerConfig(name=special_name, url="https://example.com")
        assert sse_config.name == special_name

    def test_very_large_timeout_values(self):
        """Test with very large timeout values."""
        large_timeout = 999999.999

        stdio_config = StdioServerConfig(name="test", command="python", read_timeout_seconds=large_timeout)
        assert stdio_config.read_timeout_seconds == large_timeout

        sse_config = SseServerConfig(
            name="test", url="https://example.com", timeout=large_timeout, sse_read_timeout=large_timeout * 2
        )
        assert sse_config.timeout == large_timeout
        assert sse_config.sse_read_timeout == large_timeout * 2

    def test_negative_timeout_values(self):
        """Test with negative timeout values."""
        # These should be allowed by the type system but might cause issues
        stdio_config = StdioServerConfig(name="test", command="python", read_timeout_seconds=-5.0)
        assert stdio_config.read_timeout_seconds == -5.0

        sse_config = SseServerConfig(name="test", url="https://example.com", timeout=-1.0, sse_read_timeout=-10.0)
        assert sse_config.timeout == -1.0
        assert sse_config.sse_read_timeout == -10.0

    def test_large_environment_variables(self):
        """Test with large environment variables dictionary."""
        large_env = {f"VAR_{i}": f"value_{i}" for i in range(1000)}

        config = StdioServerConfig(name="test", command="python", env=large_env)

        assert len(config.env) == 1000
        assert config.env["VAR_0"] == "value_0"
        assert config.env["VAR_999"] == "value_999"

    def test_large_headers_dictionary(self):
        """Test with large headers dictionary."""
        large_headers = {f"X-Header-{i}": f"value_{i}" for i in range(100)}

        config = SseServerConfig(name="test", url="https://example.com", headers=large_headers)

        assert len(config.headers) == 100
        assert config.headers["X-Header-0"] == "value_0"
        assert config.headers["X-Header-99"] == "value_99"

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = StdioServerConfig(name="test", command="python", args=["--version"], cache_tools=True)

        config2 = StdioServerConfig(name="test", command="python", args=["--version"], cache_tools=True)

        config3 = StdioServerConfig(name="different", command="python", args=["--version"], cache_tools=True)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test configuration string representation."""
        config = StdioServerConfig(name="test_server", command="python", args=["--version"])

        repr_str = repr(config)
        assert "StdioServerConfig" in repr_str
        assert "test_server" in repr_str
        assert "python" in repr_str

    def test_inheritance_chain(self):
        """Test that configurations properly inherit from base class."""
        stdio_config = StdioServerConfig(name="stdio", command="python")
        sse_config = SseServerConfig(name="sse", url="https://example.com")

        assert isinstance(stdio_config, MCPServerConfig)
        assert isinstance(sse_config, MCPServerConfig)

        # Both should have base class attributes
        assert hasattr(stdio_config, "name")
        assert hasattr(stdio_config, "cache_tools")
        assert hasattr(sse_config, "name")
        assert hasattr(sse_config, "cache_tools")
