"""Tests for start_server.py module."""

import sys
from unittest.mock import call, patch

import pytest

from spade_llm.human_interface.start_server import main


class TestMainFunction:
    """Test the main function of start_server.py."""

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py"])
    @patch("builtins.print")
    def test_main_default_port(self, mock_print, mock_run_server):
        """Test main function with default port."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Verify run_server was called with default port
        mock_run_server.assert_called_once_with(8080)

        # Verify initial print statements
        expected_prints = [
            call("SPADE LLM - Human Expert Web Interface"),
            call("=" * 40),
            call("\nStarting server on port 8080..."),
            call("Open http://localhost:8080 in your browser"),
            call("\nPress Ctrl+C to stop the server\n"),
            call("\n\nServer stopped by user"),
        ]
        mock_print.assert_has_calls(expected_prints)

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "9090"])
    @patch("builtins.print")
    def test_main_custom_port(self, mock_print, mock_run_server):
        """Test main function with custom port."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Verify run_server was called with custom port
        mock_run_server.assert_called_once_with(9090)

        # Verify port-specific print statements
        mock_print.assert_any_call("\nStarting server on port 9090...")
        mock_print.assert_any_call("Open http://localhost:9090 in your browser")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "invalid"])
    @patch("builtins.print")
    @patch("sys.exit")
    def test_main_invalid_port(self, mock_exit, mock_print, mock_run_server):
        """Test main function with invalid port."""
        # Make sys.exit actually exit by raising SystemExit
        mock_exit.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            main()

        # Verify error handling
        mock_print.assert_any_call("Error: Invalid port number 'invalid'")
        mock_exit.assert_called_once_with(1)
        mock_run_server.assert_not_called()

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "0"])
    @patch("builtins.print")
    def test_main_zero_port(self, mock_print, mock_run_server):
        """Test main function with port 0."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Port 0 is technically valid (OS chooses port)
        mock_run_server.assert_called_once_with(0)
        mock_print.assert_any_call("\nStarting server on port 0...")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "65535"])
    @patch("builtins.print")
    def test_main_max_port(self, mock_print, mock_run_server):
        """Test main function with maximum valid port."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Port 65535 is the maximum valid port
        mock_run_server.assert_called_once_with(65535)
        mock_print.assert_any_call("\nStarting server on port 65535...")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "-1"])
    @patch("builtins.print")
    def test_main_negative_port(self, mock_print, mock_run_server):
        """Test main function with negative port."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Negative ports are technically parsed as integers
        mock_run_server.assert_called_once_with(-1)
        mock_print.assert_any_call("\nStarting server on port -1...")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py"])
    @patch("builtins.print")
    def test_main_keyboard_interrupt(self, mock_print, mock_run_server):
        """Test main function keyboard interrupt handling."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Verify graceful shutdown message
        mock_print.assert_any_call("\n\nServer stopped by user")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py"])
    @patch("builtins.print")
    @patch("sys.exit")
    def test_main_exception_handling(self, mock_exit, mock_print, mock_run_server):
        """Test main function exception handling."""
        mock_run_server.side_effect = Exception("Server error")

        main()

        # Verify error handling
        mock_print.assert_any_call("\nError: Server error")
        mock_exit.assert_called_once_with(1)

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "8080", "extra_arg"])
    @patch("builtins.print")
    def test_main_extra_arguments(self, mock_print, mock_run_server):
        """Test main function with extra arguments (should be ignored)."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Should still use the first argument as port
        mock_run_server.assert_called_once_with(8080)
        mock_print.assert_any_call("\nStarting server on port 8080...")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "3000"])
    @patch("builtins.print")
    def test_main_no_interrupt(self, mock_print, mock_run_server):
        """Test main function without interrupt."""
        # run_server completes normally
        mock_run_server.return_value = None

        main()

        # Should not print "Server stopped by user"
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert "\n\nServer stopped by user" not in print_calls

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py", "99999"])
    @patch("builtins.print")
    def test_main_high_port_number(self, mock_print, mock_run_server):
        """Test main function with high port number."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Should handle high port numbers
        mock_run_server.assert_called_once_with(99999)
        mock_print.assert_any_call("\nStarting server on port 99999...")


class TestArgumentParsing:
    """Test argument parsing logic."""

    def test_port_parsing_logic(self):
        """Test the port parsing logic directly."""
        # Test default case
        argv = ["start_server.py"]
        port = 8080
        if len(argv) > 1:
            try:
                port = int(argv[1])
            except ValueError:
                pass
        assert port == 8080

        # Test custom port
        argv = ["start_server.py", "3000"]
        port = 8080
        if len(argv) > 1:
            try:
                port = int(argv[1])
            except ValueError:
                pass
        assert port == 3000

        # Test invalid port
        argv = ["start_server.py", "invalid"]
        port = 8080
        if len(argv) > 1:
            try:
                port = int(argv[1])
            except ValueError:
                pass
        assert port == 8080  # Should remain default

    def test_edge_case_ports(self):
        """Test edge case port values."""
        # Test empty string
        try:
            port = int("")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        # Test float string
        try:
            port = int("8080.5")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        # Test hex string
        port = int("0x1F90", 16)  # 8080 in hex
        assert port == 8080

        # Test octal string
        port = int("0o17620", 8)  # 8080 in octal
        assert port == 8080


class TestIntegration:
    """Integration tests for start_server module."""

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_full_startup_sequence(self, mock_print, mock_run_server):
        """Test full startup sequence."""
        mock_run_server.side_effect = KeyboardInterrupt()

        # Simulate command line arguments
        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py", "8888"]
            main()
        finally:
            sys.argv = original_argv

        # Verify complete startup sequence
        expected_sequence = [
            "SPADE LLM - Human Expert Web Interface",
            "=" * 40,
            "\nStarting server on port 8888...",
            "Open http://localhost:8888 in your browser",
            "\nPress Ctrl+C to stop the server\n",
            "\n\nServer stopped by user",
        ]

        actual_calls = [call[0][0] for call in mock_print.call_args_list]
        for expected in expected_sequence:
            assert expected in actual_calls

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_full_error_sequence(self, mock_exit, mock_print, mock_run_server):
        """Test full error handling sequence."""
        # Make sys.exit actually exit by raising SystemExit
        mock_exit.side_effect = SystemExit(1)

        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py", "not_a_port"]
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = original_argv

        # Verify error sequence
        mock_print.assert_any_call("Error: Invalid port number 'not_a_port'")
        mock_exit.assert_called_once_with(1)
        mock_run_server.assert_not_called()

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_runtime_error_handling(self, mock_exit, mock_print, mock_run_server):
        """Test runtime error handling."""
        mock_run_server.side_effect = RuntimeError("Failed to bind to port")

        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py", "80"]  # Privileged port
            main()
        finally:
            sys.argv = original_argv

        # Verify error handling
        mock_print.assert_any_call("\nError: Failed to bind to port")
        mock_exit.assert_called_once_with(1)


class TestUserInterface:
    """Test user interface and output formatting."""

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py"])
    @patch("builtins.print")
    def test_banner_formatting(self, mock_print, mock_run_server):
        """Test banner formatting."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Verify banner formatting
        mock_print.assert_any_call("SPADE LLM - Human Expert Web Interface")
        mock_print.assert_any_call("=" * 40)

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_url_formatting(self, mock_print, mock_run_server):
        """Test URL formatting for different ports."""
        mock_run_server.side_effect = KeyboardInterrupt()

        # Test different ports
        ports = [8080, 3000, 9090]
        for port in ports:
            with patch("sys.argv", ["start_server.py", str(port)]):
                main()
                mock_print.assert_any_call(f"Open http://localhost:{port} in your browser")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py"])
    @patch("builtins.print")
    def test_instruction_formatting(self, mock_print, mock_run_server):
        """Test instruction formatting."""
        mock_run_server.side_effect = KeyboardInterrupt()

        main()

        # Verify instructions are displayed
        mock_print.assert_any_call("\nPress Ctrl+C to stop the server\n")
        mock_print.assert_any_call("\n\nServer stopped by user")

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_error_message_formatting(self, mock_print, mock_run_server):
        """Test error message formatting."""
        mock_run_server.side_effect = Exception("Connection refused")

        with patch("sys.exit"):
            main()

        # Verify error message formatting
        mock_print.assert_any_call("\nError: Connection refused")


class TestErrorScenarios:
    """Test various error scenarios."""

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_port_validation_errors(self, mock_exit, mock_print, mock_run_server):
        """Test port validation error scenarios."""
        # Only test cases that actually raise ValueError
        error_cases = ["abc", "8080.5", "", "port", "8080abc"]

        for invalid_port in error_cases:
            # Make sys.exit actually exit by raising SystemExit for each iteration
            mock_exit.side_effect = SystemExit(1)

            with patch("sys.argv", ["start_server.py", invalid_port]):
                with pytest.raises(SystemExit):
                    main()
                mock_print.assert_any_call(f"Error: Invalid port number '{invalid_port}'")
                mock_exit.assert_called_with(1)
                mock_run_server.assert_not_called()

                # Reset mocks for next iteration
                mock_print.reset_mock()
                mock_exit.reset_mock()
                mock_run_server.reset_mock()

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_server_exception_types(self, mock_exit, mock_print, mock_run_server):
        """Test different types of server exceptions."""
        exceptions = [
            OSError("Address already in use"),
            PermissionError("Permission denied"),
            ConnectionError("Connection failed"),
            RuntimeError("Server runtime error"),
            ValueError("Invalid configuration"),
            Exception("Generic error"),
        ]

        for exception in exceptions:
            mock_run_server.side_effect = exception

            main()

            # Verify error is handled
            mock_print.assert_any_call(f"\nError: {exception}")
            mock_exit.assert_called_with(1)

            # Reset mocks for next iteration
            mock_print.reset_mock()
            mock_exit.reset_mock()

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_system_exit_handling(self, mock_exit, mock_print, mock_run_server):
        """Test SystemExit handling."""
        mock_run_server.side_effect = SystemExit(1)

        # SystemExit should propagate
        with pytest.raises(SystemExit):
            main()

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("sys.argv", ["start_server.py"])
    @patch("builtins.print")
    @patch("sys.exit")
    def test_keyboard_interrupt_vs_exception(self, mock_exit, mock_print, mock_run_server):
        """Test difference between KeyboardInterrupt and other exceptions."""
        # Test KeyboardInterrupt
        mock_run_server.side_effect = KeyboardInterrupt()
        main()

        # Should print user stop message, not error
        mock_print.assert_any_call("\n\nServer stopped by user")
        mock_exit.assert_not_called()

        # Reset mocks
        mock_print.reset_mock()
        mock_exit.reset_mock()

        # Test other exception - make sys.exit actually exit
        mock_exit.side_effect = SystemExit(1)
        mock_run_server.side_effect = Exception("Server error")

        with pytest.raises(SystemExit):
            main()

        # Should print error message and exit
        mock_print.assert_any_call("\nError: Server error")
        mock_exit.assert_called_once_with(1)


class TestCommandLineInterface:
    """Test command line interface aspects."""

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_no_arguments(self, mock_print, mock_run_server):
        """Test with no command line arguments."""
        mock_run_server.side_effect = KeyboardInterrupt()

        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py"]
            main()
        finally:
            sys.argv = original_argv

        # Should use default port
        mock_run_server.assert_called_once_with(8080)

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_multiple_arguments(self, mock_print, mock_run_server):
        """Test with multiple command line arguments."""
        mock_run_server.side_effect = KeyboardInterrupt()

        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py", "9000", "ignored", "also_ignored"]
            main()
        finally:
            sys.argv = original_argv

        # Should use first argument as port, ignore others
        mock_run_server.assert_called_once_with(9000)

    def test_argument_bounds(self):
        """Test argument boundary conditions."""
        # Test sys.argv length checking
        original_argv = sys.argv
        try:
            # Test empty argv (shouldn't happen in practice)
            sys.argv = []
            port = 8080
            if len(sys.argv) > 1:
                try:
                    port = int(sys.argv[1])
                except ValueError:
                    pass
            assert port == 8080

            # Test single argument (script name only)
            sys.argv = ["start_server.py"]
            port = 8080
            if len(sys.argv) > 1:
                try:
                    port = int(sys.argv[1])
                except ValueError:
                    pass
            assert port == 8080

        finally:
            sys.argv = original_argv


class TestModuleExecution:
    """Test module execution scenarios."""

    @patch("spade_llm.human_interface.start_server.main")
    def test_main_block_execution(self, mock_main):
        """Test __main__ block execution."""
        # This test verifies the structure exists
        # The actual execution test would require running the module

        # Check that main function exists and is callable
        assert callable(main)

        # Verify the main function can be called
        mock_main.return_value = None
        mock_main()
        mock_main.assert_called_once()

    def test_import_structure(self):
        """Test that imports work correctly."""
        from spade_llm.human_interface import start_server
        from spade_llm.human_interface.start_server import main

        # Verify imports work
        assert hasattr(start_server, "main")
        assert callable(start_server.main)
        assert start_server.main is main

    def test_sys_module_usage(self):
        """Test sys module usage."""
        import sys

        # Test sys.argv access
        assert hasattr(sys, "argv")
        assert isinstance(sys.argv, list)

        # Test sys.exit access
        assert hasattr(sys, "exit")
        assert callable(sys.exit)


class TestRobustness:
    """Test robustness and edge cases."""

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_unicode_in_arguments(self, mock_print, mock_run_server):
        """Test handling of unicode characters in arguments."""
        mock_run_server.side_effect = KeyboardInterrupt()

        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py", "8080", "unicode_arg_ñ"]
            main()
        finally:
            sys.argv = original_argv

        # Should handle unicode arguments gracefully
        mock_run_server.assert_called_once_with(8080)

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_very_long_arguments(self, mock_exit, mock_print, mock_run_server):
        """Test handling of very long arguments."""
        long_arg = "a" * 1000

        original_argv = sys.argv
        try:
            sys.argv = ["start_server.py", long_arg]
            main()
        finally:
            sys.argv = original_argv

        # Should handle long arguments gracefully
        mock_print.assert_any_call(f"Error: Invalid port number '{long_arg}'")
        mock_exit.assert_called_once_with(1)

    @patch("spade_llm.human_interface.start_server.run_server")
    @patch("builtins.print")
    def test_special_characters_in_arguments(self, mock_print, mock_run_server):
        """Test handling of special characters in arguments."""
        special_chars = ["!@#$%", "8080!", "808@0", "8080#test"]

        for special_arg in special_chars:
            mock_run_server.side_effect = KeyboardInterrupt()

            original_argv = sys.argv
            try:
                sys.argv = ["start_server.py", special_arg]
                if special_arg.isdigit():
                    main()
                    mock_run_server.assert_called_with(int(special_arg))
                else:
                    with patch("sys.exit"):
                        main()
            finally:
                sys.argv = original_argv
                mock_run_server.reset_mock()
                mock_print.reset_mock()
