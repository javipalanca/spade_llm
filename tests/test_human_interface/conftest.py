"""Configuration and fixtures for human_interface tests."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_http_server():
    """Mock HTTPServer for testing."""
    with patch("spade_llm.human_interface.web_server.HTTPServer") as mock_server:
        mock_instance = Mock()
        mock_server.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_os_operations():
    """Mock OS operations for testing."""
    with (
        patch("spade_llm.human_interface.web_server.os.makedirs") as mock_makedirs,
        patch("spade_llm.human_interface.web_server.os.chdir") as mock_chdir,
        patch("spade_llm.human_interface.web_server.os.path.dirname") as mock_dirname,
        patch("spade_llm.human_interface.web_server.os.path.abspath") as mock_abspath,
        patch("spade_llm.human_interface.web_server.os.path.join") as mock_join,
    ):
        # Set up default return values
        mock_abspath.return_value = "/mock/path/to/web_server.py"
        mock_dirname.return_value = "/mock/path/to"
        mock_join.return_value = "/mock/path/to/web_client"

        yield {
            "makedirs": mock_makedirs,
            "chdir": mock_chdir,
            "dirname": mock_dirname,
            "abspath": mock_abspath,
            "join": mock_join,
        }


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch("spade_llm.human_interface.web_server.logger") as mock_log:
        yield mock_log


@pytest.fixture
def mock_print():
    """Mock print function for testing."""
    with patch("builtins.print") as mock_print_func:
        yield mock_print_func


@pytest.fixture
def mock_sys_exit():
    """Mock sys.exit for testing."""
    with patch("sys.exit") as mock_exit:
        yield mock_exit


@pytest.fixture
def mock_run_server():
    """Mock run_server function for testing."""
    with patch("spade_llm.human_interface.start_server.run_server") as mock_run:
        yield mock_run


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_cors_handler():
    """Mock CORS handler for testing."""
    mock_request = Mock()
    mock_client_address = ("127.0.0.1", 12345)
    mock_server = Mock()

    # Mock the parent class methods
    with patch("spade_llm.human_interface.web_server.SimpleHTTPRequestHandler.__init__", return_value=None):
        from spade_llm.human_interface.web_server import CORSRequestHandler

        handler = CORSRequestHandler(mock_request, mock_client_address, mock_server)
        handler.send_header = Mock()
        handler.send_response = Mock()
        yield handler


@pytest.fixture
def mock_argv():
    """Mock sys.argv for testing."""
    original_argv = os.sys.argv
    yield
    os.sys.argv = original_argv


@pytest.fixture
def sample_ports():
    """Sample port numbers for testing."""
    return [8080, 3000, 9090, 8888, 5000]


@pytest.fixture
def invalid_ports():
    """Invalid port strings for testing."""
    return ["invalid", "abc", "8080.5", "", "port", "8080abc", "8080 ", " 8080"]


@pytest.fixture
def valid_ports():
    """Valid port numbers for testing."""
    return [80, 443, 8080, 3000, 9090, 65535]


@pytest.fixture
def exception_types():
    """Common exception types for testing."""
    return [
        OSError("Address already in use"),
        PermissionError("Permission denied"),
        ConnectionError("Connection failed"),
        RuntimeError("Server runtime error"),
        ValueError("Invalid configuration"),
        Exception("Generic error"),
    ]


@pytest.fixture(autouse=True)
def reset_mocks():
    """Automatically reset mocks after each test."""
    yield
    # Any cleanup code can go here if needed


class MockRequestHandler:
    """Mock request handler for testing."""

    def __init__(self):
        self.headers_sent = []
        self.responses_sent = []

    def send_header(self, name, value):
        self.headers_sent.append((name, value))

    def send_response(self, code):
        self.responses_sent.append(code)

    def end_headers(self):
        pass


@pytest.fixture
def mock_request_handler():
    """Mock request handler instance."""
    return MockRequestHandler()


@pytest.fixture
def server_config():
    """Server configuration for testing."""
    return {
        "host": "localhost",
        "default_port": 8080,
        "max_port": 65535,
        "min_port": 1,
        "default_directory": "web_client",
    }


@pytest.fixture
def cors_headers():
    """Expected CORS headers for testing."""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Cache-Control": "no-store, no-cache, must-revalidate",
    }


@pytest.fixture
def mock_keyboard_interrupt():
    """Mock keyboard interrupt for testing."""
    return KeyboardInterrupt()


@pytest.fixture
def mock_server_error():
    """Mock server error for testing."""
    return OSError("Server error")


@pytest.fixture
def command_line_scenarios():
    """Command line test scenarios."""
    return [
        {"argv": ["start_server.py"], "expected_port": 8080, "should_error": False},
        {"argv": ["start_server.py", "9090"], "expected_port": 9090, "should_error": False},
        {"argv": ["start_server.py", "invalid"], "expected_port": None, "should_error": True},
        {"argv": ["start_server.py", "65535"], "expected_port": 65535, "should_error": False},
        {"argv": ["start_server.py", "0"], "expected_port": 0, "should_error": False},
    ]


@pytest.fixture
def web_server_test_cases():
    """Web server test cases."""
    return [
        {
            "name": "default_config",
            "port": 8080,
            "directory": None,
            "expected_host": "localhost",
            "expected_port": 8080,
        },
        {"name": "custom_port", "port": 9090, "directory": None, "expected_host": "localhost", "expected_port": 9090},
        {
            "name": "custom_directory",
            "port": 8080,
            "directory": "/custom/path",
            "expected_host": "localhost",
            "expected_port": 8080,
        },
        {
            "name": "custom_all",
            "port": 3000,
            "directory": "/test/dir",
            "expected_host": "localhost",
            "expected_port": 3000,
        },
    ]


@pytest.fixture
def error_scenarios():
    """Error scenarios for testing."""
    return [
        {"name": "makedirs_error", "mock_target": "makedirs", "exception": OSError("Permission denied")},
        {"name": "chdir_error", "mock_target": "chdir", "exception": OSError("Directory not found")},
        {"name": "server_creation_error", "mock_target": "HTTPServer", "exception": OSError("Address already in use")},
        {"name": "permission_error", "mock_target": "makedirs", "exception": PermissionError("Permission denied")},
    ]


# Helper functions for tests
def create_mock_server(serve_forever_side_effect=None):
    """Create a mock server with specified behavior."""
    mock_server = Mock()
    if serve_forever_side_effect:
        mock_server.serve_forever.side_effect = serve_forever_side_effect
    return mock_server


def assert_cors_headers(handler, expected_headers):
    """Assert that CORS headers are set correctly."""
    calls = handler.send_header.call_args_list
    actual_headers = {call[0][0]: call[0][1] for call in calls}

    for header, expected_value in expected_headers.items():
        assert header in actual_headers
        assert actual_headers[header] == expected_value


def assert_print_sequence(mock_print, expected_sequence):
    """Assert that print calls match expected sequence."""
    actual_calls = [call[0][0] for call in mock_print.call_args_list]

    for expected in expected_sequence:
        assert expected in actual_calls


def create_test_argv(port=None, extra_args=None):
    """Create test sys.argv with optional port and extra arguments."""
    argv = ["start_server.py"]

    if port is not None:
        argv.append(str(port))

    if extra_args:
        argv.extend(extra_args)

    return argv


# Test data
TEST_PORTS = [80, 443, 8080, 3000, 9090, 65535]
INVALID_PORTS = ["invalid", "abc", "8080.5", "", "port", "8080abc"]
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Cache-Control": "no-store, no-cache, must-revalidate",
}
EXPECTED_PRINT_SEQUENCE = [
    "SPADE LLM - Human Expert Web Interface",
    "=" * 40,
    "\nStarting server on port 8080...",
    "Open http://localhost:8080 in your browser",
    "\nPress Ctrl+C to stop the server\n",
]
