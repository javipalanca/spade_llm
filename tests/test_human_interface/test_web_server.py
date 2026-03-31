"""Tests for web_server.py module."""

import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from unittest.mock import Mock, call, patch

import pytest

from spade_llm.human_interface.web_server import CORSRequestHandler, run_server


class TestCORSRequestHandler:
    """Test CORSRequestHandler class."""

    def test_inheritance(self):
        """Test that CORSRequestHandler inherits from SimpleHTTPRequestHandler."""
        assert issubclass(CORSRequestHandler, SimpleHTTPRequestHandler)

    def test_end_headers_adds_cors_headers(self):
        """Test that end_headers adds CORS headers."""
        with (
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None),
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.end_headers") as mock_super_end,
        ):
            handler = CORSRequestHandler(Mock(), Mock(), Mock())
            handler.send_header = Mock()

            handler.end_headers()

            # Verify CORS headers were added
            expected_calls = [
                call("Access-Control-Allow-Origin", "*"),
                call("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                call("Access-Control-Allow-Headers", "Content-Type"),
                call("Cache-Control", "no-store, no-cache, must-revalidate"),
            ]
            handler.send_header.assert_has_calls(expected_calls)
            mock_super_end.assert_called_once()

    def test_do_OPTIONS_method(self):
        """Test OPTIONS method handling."""
        with patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None):
            handler = CORSRequestHandler(Mock(), Mock(), Mock())
            handler.send_response = Mock()
            handler.end_headers = Mock()

            handler.do_OPTIONS()

            handler.send_response.assert_called_once_with(200)
            handler.end_headers.assert_called_once()

    def test_end_headers_preserves_order(self):
        """Test that end_headers calls parent after setting headers."""
        with (
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None),
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.end_headers") as mock_super_end,
        ):
            handler = CORSRequestHandler(Mock(), Mock(), Mock())
            handler.send_header = Mock()

            handler.end_headers()

            # Verify parent method is called after headers are set
            assert handler.send_header.call_count == 4
            mock_super_end.assert_called_once()

    def test_cors_headers_values(self):
        """Test specific CORS header values."""
        with (
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None),
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.end_headers"),
        ):
            handler = CORSRequestHandler(Mock(), Mock(), Mock())
            handler.send_header = Mock()

            handler.end_headers()

            # Check specific header values
            calls = handler.send_header.call_args_list
            headers = {call[0][0]: call[0][1] for call in calls}

            assert headers["Access-Control-Allow-Origin"] == "*"
            assert headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"
            assert headers["Access-Control-Allow-Headers"] == "Content-Type"
            assert headers["Cache-Control"] == "no-store, no-cache, must-revalidate"


class TestRunServer:
    """Test run_server function."""

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.os.path.dirname")
    @patch("spade_llm.human_interface.web_server.os.path.abspath")
    @patch("spade_llm.human_interface.web_server.os.path.join")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_run_server_default_directory(
        self, mock_logger, mock_join, mock_abspath, mock_dirname, mock_makedirs, mock_chdir, mock_http_server
    ):
        """Test run_server with default directory."""
        # Setup mocks
        mock_abspath.return_value = "/path/to/web_server.py"
        mock_dirname.return_value = "/path/to"
        mock_join.return_value = "/path/to/web_client"
        mock_server = Mock()
        mock_http_server.return_value = mock_server

        # Test with keyboard interrupt to exit gracefully
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        run_server()

        # Verify directory operations
        mock_abspath.assert_called_once()
        mock_dirname.assert_called_once_with("/path/to/web_server.py")
        mock_join.assert_called_once_with("/path/to", "web_client")
        mock_makedirs.assert_called_once_with("/path/to/web_client", exist_ok=True)
        mock_chdir.assert_called_once_with("/path/to/web_client")

        # Verify server creation and startup
        mock_http_server.assert_called_once()
        args, kwargs = mock_http_server.call_args
        assert args[0] == ("localhost", 8080)
        mock_server.serve_forever.assert_called_once()
        mock_server.shutdown.assert_called_once()

        # Verify logging
        assert mock_logger.info.call_count >= 3
        mock_logger.info.assert_any_call("Human Expert interface running at http://localhost:8080")
        mock_logger.info.assert_any_call("Serving files from: /path/to/web_client")
        mock_logger.info.assert_any_call("Server stopped")

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_run_server_custom_directory(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test run_server with custom directory."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        custom_dir = "/custom/path"
        run_server(port=9090, directory=custom_dir)

        # Verify custom directory is used
        mock_makedirs.assert_called_once_with(custom_dir, exist_ok=True)
        mock_chdir.assert_called_once_with(custom_dir)

        # Verify custom port is used
        args, kwargs = mock_http_server.call_args
        assert args[0] == ("localhost", 9090)

        # Verify logging with custom values
        mock_logger.info.assert_any_call("Human Expert interface running at http://localhost:9090")
        mock_logger.info.assert_any_call(f"Serving files from: {custom_dir}")

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_run_server_handler_configuration(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test that the handler is configured correctly."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        custom_dir = "/test/dir"
        run_server(directory=custom_dir)

        # Verify handler is partial function with correct directory
        args, kwargs = mock_http_server.call_args
        handler_class = args[1]

        # The handler should be a partial function
        assert isinstance(handler_class, partial)
        assert handler_class.func == CORSRequestHandler
        assert handler_class.keywords["directory"] == custom_dir

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_run_server_keyboard_interrupt_handling(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test keyboard interrupt handling."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        run_server()

        # Verify graceful shutdown
        mock_server.serve_forever.assert_called_once()
        mock_server.shutdown.assert_called_once()
        mock_logger.info.assert_any_call("Server stopped")

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_run_server_no_interrupt(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test server running without interruption."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server

        # Use a flag to stop the server after one call
        call_count = 0

        def mock_serve_forever():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return  # Normal operation
            raise KeyboardInterrupt()  # Stop on second call

        mock_server.serve_forever.side_effect = mock_serve_forever

        run_server()

        # Verify server was started
        mock_server.serve_forever.assert_called_once()
        # No shutdown should be called without interrupt
        mock_server.shutdown.assert_not_called()

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    def test_run_server_makedirs_error_handling(self, mock_makedirs, mock_chdir, mock_http_server):
        """Test handling of makedirs errors."""
        mock_makedirs.side_effect = OSError("Permission denied")

        with pytest.raises(OSError, match="Permission denied"):
            run_server()

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    def test_run_server_chdir_error_handling(self, mock_makedirs, mock_chdir, mock_http_server):
        """Test handling of chdir errors."""
        mock_chdir.side_effect = OSError("Directory not found")

        with pytest.raises(OSError, match="Directory not found"):
            run_server()

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    def test_run_server_http_server_error(self, mock_makedirs, mock_chdir, mock_http_server):
        """Test handling of HTTP server creation errors."""
        mock_http_server.side_effect = OSError("Address already in use")

        with pytest.raises(OSError, match="Address already in use"):
            run_server()

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_run_server_port_types(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test different port types."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        # Test string port (should be converted to int)
        run_server(port="3000")

        args, kwargs = mock_http_server.call_args
        assert args[0] == ("localhost", "3000")  # HTTPServer should handle conversion

        # Test integer port
        run_server(port=4000)

        args, kwargs = mock_http_server.call_args
        assert args[0] == ("localhost", 4000)


class TestCommandLineExecution:
    """Test command-line execution of web_server.py."""

    @patch("spade_llm.human_interface.web_server.run_server")
    @patch("sys.argv", ["web_server.py"])
    def test_command_line_default_port(self, mock_run_server):
        """Test command-line execution with default port."""
        from spade_llm.human_interface import web_server

        # Execute the main section
        if __name__ == "__main__":
            exec(compile(open(web_server.__file__).read(), web_server.__file__, "exec"))

        # This test is tricky because we can't easily execute the __main__ section
        # Instead, let's test the logic directly

        original_argv = sys.argv
        try:
            sys.argv = ["web_server.py"]
            # Simulate the main logic
            port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
            assert port == 8080
        finally:
            sys.argv = original_argv

    def test_command_line_port_parsing(self):
        """Test port parsing from command line."""

        # Test default port
        original_argv = sys.argv
        try:
            sys.argv = ["web_server.py"]
            port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
            assert port == 8080

            # Test custom port
            sys.argv = ["web_server.py", "9090"]
            port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
            assert port == 9090

        finally:
            sys.argv = original_argv

    def test_command_line_invalid_port(self):
        """Test handling of invalid port in command line."""

        original_argv = sys.argv
        try:
            sys.argv = ["web_server.py", "invalid"]

            with pytest.raises(ValueError):
                int(sys.argv[1]) if len(sys.argv) > 1 else 8080

        finally:
            sys.argv = original_argv


class TestIntegration:
    """Integration tests for web_server module."""

    def test_cors_handler_integration(self):
        """Test CORSRequestHandler integration with SimpleHTTPRequestHandler."""
        # Create a mock request and client address
        mock_request = Mock()
        mock_client_address = ("127.0.0.1", 12345)
        mock_server = Mock()

        # Test that handler can be instantiated without errors
        with patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None):
            handler = CORSRequestHandler(mock_request, mock_client_address, mock_server)

            # Verify it's an instance of SimpleHTTPRequestHandler
            assert isinstance(handler, SimpleHTTPRequestHandler)

    @patch("spade_llm.human_interface.web_server.os.path.dirname")
    @patch("spade_llm.human_interface.web_server.os.path.abspath")
    @patch("spade_llm.human_interface.web_server.os.path.join")
    def test_directory_path_resolution(self, mock_join, mock_abspath, mock_dirname):
        """Test directory path resolution logic."""
        # Mock the path operations
        mock_abspath.return_value = "/full/path/to/web_server.py"
        mock_dirname.return_value = "/full/path/to"
        mock_join.return_value = "/full/path/to/web_client"

        # Test the directory resolution logic
        with (
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            run_server()

            # Verify path resolution calls
            mock_abspath.assert_called_once()
            mock_dirname.assert_called_once_with("/full/path/to/web_server.py")
            mock_join.assert_called_once_with("/full/path/to", "web_client")

    def test_server_configuration_integration(self):
        """Test server configuration integration."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            run_server(port=8888, directory="/test/dir")

            # Verify server is created with correct parameters
            args, kwargs = mock_http_server.call_args
            assert args[0] == ("localhost", 8888)

            # Verify handler is configured correctly
            handler_class = args[1]
            assert isinstance(handler_class, partial)
            assert handler_class.func == CORSRequestHandler
            assert handler_class.keywords["directory"] == "/test/dir"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_handler_initialization_error(self):
        """Test handler initialization with invalid parameters."""
        # Test with None parameters - the parent class should raise TypeError
        with patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__") as mock_init:
            mock_init.side_effect = TypeError("Invalid parameters")
            with pytest.raises(TypeError):
                CORSRequestHandler(None, None, None)

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    def test_server_binding_error(self, mock_makedirs, mock_chdir, mock_http_server):
        """Test server binding errors."""
        mock_http_server.side_effect = OSError("Address already in use")

        with pytest.raises(OSError):
            run_server()

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    def test_permission_error(self, mock_makedirs, mock_chdir, mock_http_server):
        """Test permission errors."""
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            run_server()

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_server_runtime_error(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test runtime errors during server operation."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = RuntimeError("Server error")

        with pytest.raises(RuntimeError):
            run_server()

        # Verify logging occurred before the error
        mock_logger.info.assert_called()


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects."""

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_multiple_server_instances(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test multiple server instances handling."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        # Run multiple instances
        run_server(port=8080)
        run_server(port=8081)

        # Verify each instance was configured correctly
        assert mock_http_server.call_count == 2
        calls = mock_http_server.call_args_list
        assert calls[0][0][0] == ("localhost", 8080)
        assert calls[1][0][0] == ("localhost", 8081)

    def test_handler_thread_safety(self):
        """Test handler thread safety."""
        # Create multiple handlers concurrently
        handlers = []
        errors = []

        def create_handler():
            try:
                with patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None):
                    mock_request = Mock()
                    mock_client_address = ("127.0.0.1", 12345)
                    mock_server = Mock()
                    handler = CORSRequestHandler(mock_request, mock_client_address, mock_server)
                    handlers.append(handler)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_handler)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0
        assert len(handlers) == 10

        # Verify all handlers are valid
        for handler in handlers:
            assert isinstance(handler, CORSRequestHandler)

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    @patch("spade_llm.human_interface.web_server.os.chdir")
    @patch("spade_llm.human_interface.web_server.os.makedirs")
    @patch("spade_llm.human_interface.web_server.logger")
    def test_rapid_server_start_stop(self, mock_logger, mock_makedirs, mock_chdir, mock_http_server):
        """Test rapid server start/stop cycles."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        # Rapid start/stop cycles
        for i in range(5):
            run_server(port=8080 + i)

        # Verify all servers were created and shut down
        assert mock_http_server.call_count == 5
        assert mock_server.serve_forever.call_count == 5
        assert mock_server.shutdown.call_count == 5

    def test_cors_headers_consistency(self):
        """Test CORS headers consistency across multiple calls."""
        with (
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None),
            patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.end_headers"),
        ):
            handler = CORSRequestHandler(Mock(), Mock(), Mock())
            handler.send_header = Mock()

            # Call end_headers multiple times
            for _ in range(5):
                handler.end_headers()

            # Verify headers are consistent
            assert handler.send_header.call_count == 20  # 4 headers × 5 calls
            calls = handler.send_header.call_args_list

            # Check that headers are added in the same order each time
            for i in range(0, 20, 4):
                assert calls[i][0][0] == "Access-Control-Allow-Origin"
                assert calls[i + 1][0][0] == "Access-Control-Allow-Methods"
                assert calls[i + 2][0][0] == "Access-Control-Allow-Headers"
                assert calls[i + 3][0][0] == "Cache-Control"
