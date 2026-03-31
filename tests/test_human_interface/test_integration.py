"""Integration tests for human_interface module."""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from spade_llm.human_interface.start_server import main
from spade_llm.human_interface.web_server import CORSRequestHandler, run_server


class TestWebServerIntegration:
    """Integration tests for web server functionality."""

    @patch("spade_llm.human_interface.web_server.HTTPServer")
    def test_server_lifecycle_integration(self, mock_http_server):
        """Test complete server lifecycle."""
        # Create a mock server that simulates real behavior
        mock_server = Mock()
        mock_http_server.return_value = mock_server

        # Test normal startup and shutdown
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        with (
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
        ):
            run_server(port=8080, directory="/test/dir")

            # Verify server lifecycle
            mock_http_server.assert_called_once()
            mock_server.serve_forever.assert_called_once()
            mock_server.shutdown.assert_called_once()

    def test_handler_configuration_integration(self):
        """Test handler configuration integration."""
        from functools import partial

        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            test_directory = "/integration/test"
            run_server(directory=test_directory)

            # Verify handler configuration
            args, kwargs = mock_http_server.call_args
            handler_class = args[1]

            assert isinstance(handler_class, partial)
            assert handler_class.func == CORSRequestHandler
            assert handler_class.keywords["directory"] == test_directory


class TestStartServerIntegration:
    """Integration tests for start server functionality."""

    @patch("spade_llm.human_interface.start_server.run_server")
    def test_main_integration_with_web_server(self, mock_run_server):
        """Test main function integration with web server."""
        mock_run_server.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["start_server.py", "9999"]), patch("builtins.print") as mock_print:
            main()

            # Verify integration
            mock_run_server.assert_called_once_with(9999)
            mock_print.assert_any_call("Open http://localhost:9999 in your browser")

    @patch("spade_llm.human_interface.start_server.run_server")
    def test_error_propagation_integration(self, mock_run_server):
        """Test error propagation between components."""
        server_error = OSError("Port already in use")
        mock_run_server.side_effect = server_error

        with (
            patch("sys.argv", ["start_server.py", "80"]),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            main()

            # Verify error propagation
            mock_print.assert_any_call("\nError: Port already in use")
            mock_exit.assert_called_once_with(1)

    def test_command_line_to_server_integration(self):
        """Test command line argument processing to server startup."""
        with patch("spade_llm.human_interface.start_server.run_server") as mock_run_server:
            mock_run_server.side_effect = KeyboardInterrupt()

            # Test various command line scenarios
            test_cases = [
                (["start_server.py"], 8080),
                (["start_server.py", "3000"], 3000),
                (["start_server.py", "9090"], 9090),
            ]

            for argv, expected_port in test_cases:
                with patch("sys.argv", argv), patch("builtins.print"):
                    main()
                    mock_run_server.assert_called_with(expected_port)
                    mock_run_server.reset_mock()


class TestFullSystemIntegration:
    """Test full system integration scenarios."""

    def test_complete_startup_sequence(self):
        """Test complete startup sequence from command line to server."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
            patch("sys.argv", ["start_server.py", "8888"]),
            patch("builtins.print") as mock_print,
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            main()

            # Verify complete sequence
            mock_http_server.assert_called_once()
            args, kwargs = mock_http_server.call_args
            assert args[0] == ("localhost", 8888)

            # Verify UI messages
            mock_print.assert_any_call("SPADE LLM - Human Expert Web Interface")
            mock_print.assert_any_call("\nStarting server on port 8888...")
            mock_print.assert_any_call("Open http://localhost:8888 in your browser")

    def test_error_handling_integration(self):
        """Test error handling across all components."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("sys.argv", ["start_server.py", "443"]),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            # Simulate server creation failure
            mock_http_server.side_effect = PermissionError("Permission denied")

            main()

            # Verify error handling
            mock_print.assert_any_call("\nError: Permission denied")
            mock_exit.assert_called_once_with(1)

    def test_directory_handling_integration(self):
        """Test directory handling across components."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs") as mock_makedirs,
            patch("spade_llm.human_interface.web_server.os.chdir") as mock_chdir,
            patch("spade_llm.human_interface.web_server.os.path.dirname") as mock_dirname,
            patch("spade_llm.human_interface.web_server.os.path.abspath") as mock_abspath,
            patch("spade_llm.human_interface.web_server.os.path.join") as mock_join,
            patch("spade_llm.human_interface.web_server.logger"),
            patch("sys.argv", ["start_server.py"]),
            patch("builtins.print"),
        ):
            # Set up path mocks
            mock_abspath.return_value = "/app/web_server.py"
            mock_dirname.return_value = "/app"
            mock_join.return_value = "/app/web_client"

            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            main()

            # Verify directory operations
            mock_makedirs.assert_called_once_with("/app/web_client", exist_ok=True)
            mock_chdir.assert_called_once_with("/app/web_client")


class TestConcurrencyIntegration:
    """Test concurrency and threading integration."""

    def test_multiple_handler_instances(self):
        """Test multiple handler instances."""
        handlers = []

        def create_handler():
            with patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None):
                handler = CORSRequestHandler(Mock(), ("127.0.0.1", 12345), Mock())
                handler.send_header = Mock()
                handlers.append(handler)

        # Create multiple handlers concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_handler)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all handlers were created
        assert len(handlers) == 5
        for handler in handlers:
            assert isinstance(handler, CORSRequestHandler)

    def test_rapid_server_operations(self):
        """Test rapid server start/stop operations."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            # Rapid operations
            for i in range(3):
                run_server(port=8080 + i)

            # Verify all operations completed
            assert mock_http_server.call_count == 3
            assert mock_server.serve_forever.call_count == 3
            assert mock_server.shutdown.call_count == 3


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_development_server_scenario(self):
        """Test typical development server usage."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger") as mock_logger,
            patch("sys.argv", ["start_server.py", "3000"]),
            patch("builtins.print") as mock_print,
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            main()

            # Verify development server setup
            mock_logger.info.assert_any_call("Human Expert interface running at http://localhost:3000")
            mock_print.assert_any_call("Open http://localhost:3000 in your browser")

    def test_production_server_scenario(self):
        """Test production-like server scenario."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
            patch("sys.argv", ["start_server.py", "80"]),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            # Simulate permission error for privileged port
            mock_http_server.side_effect = PermissionError("Permission denied")

            main()

            # Verify production error handling
            mock_print.assert_any_call("\nError: Permission denied")
            mock_exit.assert_called_once_with(1)

    def test_custom_directory_scenario(self):
        """Test custom directory scenario."""
        custom_dir = "/custom/web/content"

        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs") as mock_makedirs,
            patch("spade_llm.human_interface.web_server.os.chdir") as mock_chdir,
            patch("spade_llm.human_interface.web_server.logger") as mock_logger,
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            run_server(directory=custom_dir)

            # Verify custom directory handling
            mock_makedirs.assert_called_once_with(custom_dir, exist_ok=True)
            mock_chdir.assert_called_once_with(custom_dir)
            mock_logger.info.assert_any_call(f"Serving files from: {custom_dir}")

    def test_port_already_in_use_scenario(self):
        """Test port already in use scenario."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("sys.argv", ["start_server.py", "8080"]),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            mock_http_server.side_effect = OSError("Address already in use")

            main()

            # Verify port conflict handling
            mock_print.assert_any_call("\nError: Address already in use")
            mock_exit.assert_called_once_with(1)


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience."""

    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        with (
            patch("spade_llm.human_interface.web_server.os.makedirs") as mock_makedirs,
            patch("spade_llm.human_interface.web_server.os.chdir") as mock_chdir,
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
        ):
            # First call fails, second succeeds
            mock_makedirs.side_effect = [OSError("First failure"), None]
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            # First attempt should fail
            with pytest.raises(OSError):
                run_server()

            # Second attempt should succeed
            run_server()

            # Verify recovery
            assert mock_makedirs.call_count == 2
            assert mock_chdir.call_count == 1  # Only called on success

    def test_graceful_shutdown_integration(self):
        """Test graceful shutdown integration."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger") as mock_logger,
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            run_server()

            # Verify graceful shutdown
            mock_server.shutdown.assert_called_once()
            mock_logger.info.assert_any_call("Server stopped")

    def test_exception_chain_integration(self):
        """Test exception chain handling."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("sys.argv", ["start_server.py", "8080"]),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            # Chain of exceptions
            original_error = ConnectionError("Network error")
            wrapper_error = OSError("Server error")
            wrapper_error.__cause__ = original_error

            mock_http_server.side_effect = wrapper_error

            main()

            # Verify exception handling
            mock_print.assert_any_call("\nError: Server error")
            mock_exit.assert_called_once_with(1)


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_server_startup_time(self):
        """Test server startup time integration."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
        ):
            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            start_time = time.time()
            run_server()
            startup_time = time.time() - start_time

            # Verify reasonable startup time (should be very fast with mocks)
            assert startup_time < 1.0

    def test_memory_usage_pattern(self):
        """Test memory usage pattern."""
        # Create and destroy multiple server instances
        servers = []

        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.logger"),
        ):
            for i in range(5):
                mock_server = Mock()
                mock_http_server.return_value = mock_server
                mock_server.serve_forever.side_effect = KeyboardInterrupt()
                servers.append(mock_server)

                run_server(port=8080 + i)

            # Verify all servers were created and cleaned up
            assert len(servers) == 5
            for server in servers:
                server.shutdown.assert_called_once()


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    def test_environment_based_configuration(self):
        """Test environment-based configuration."""
        with patch.dict("os.environ", {"SPADE_LLM_PORT": "9999"}):
            # Note: The current implementation doesn't use environment variables
            # This test demonstrates how it could be extended
            with patch("spade_llm.human_interface.start_server.run_server") as mock_run_server:
                mock_run_server.side_effect = KeyboardInterrupt()

                with patch("sys.argv", ["start_server.py"]), patch("builtins.print"):
                    main()

                    # Currently uses default port, but could be extended
                    mock_run_server.assert_called_once_with(8080)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test various configuration scenarios
        configs = [
            {"port": 8080, "directory": "/valid/path"},
            {"port": 3000, "directory": None},
            {"port": 9090, "directory": "/custom/dir"},
        ]

        for config in configs:
            with (
                patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
                patch("spade_llm.human_interface.web_server.os.makedirs"),
                patch("spade_llm.human_interface.web_server.os.chdir"),
                patch("spade_llm.human_interface.web_server.logger"),
            ):
                mock_server = Mock()
                mock_http_server.return_value = mock_server
                mock_server.serve_forever.side_effect = KeyboardInterrupt()

                run_server(port=config["port"], directory=config["directory"])

                # Verify configuration was applied
                args, kwargs = mock_http_server.call_args
                assert args[0] == ("localhost", config["port"])

    def test_default_configuration_integration(self):
        """Test default configuration integration."""
        with (
            patch("spade_llm.human_interface.web_server.HTTPServer") as mock_http_server,
            patch("spade_llm.human_interface.web_server.os.makedirs"),
            patch("spade_llm.human_interface.web_server.os.chdir"),
            patch("spade_llm.human_interface.web_server.os.path.dirname") as mock_dirname,
            patch("spade_llm.human_interface.web_server.os.path.abspath") as mock_abspath,
            patch("spade_llm.human_interface.web_server.os.path.join") as mock_join,
            patch("spade_llm.human_interface.web_server.logger"),
        ):
            # Set up default path resolution
            mock_abspath.return_value = "/app/web_server.py"
            mock_dirname.return_value = "/app"
            mock_join.return_value = "/app/web_client"

            mock_server = Mock()
            mock_http_server.return_value = mock_server
            mock_server.serve_forever.side_effect = KeyboardInterrupt()

            run_server()  # Use all defaults

            # Verify default configuration
            args, kwargs = mock_http_server.call_args
            assert args[0] == ("localhost", 8080)

            # Verify default directory resolution
            mock_join.assert_called_once_with("/app", "web_client")
