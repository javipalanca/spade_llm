"""Tests for routing types."""

from unittest.mock import Mock

import pytest
from spade.message import Message

from spade_llm.routing.types import RoutingFunction, RoutingResponse


class TestRoutingResponse:
    """Test RoutingResponse dataclass."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        response = RoutingResponse(recipients="test@localhost")

        assert response.recipients == "test@localhost"
        assert response.transform is None
        assert response.metadata is None

    def test_init_single_recipient(self):
        """Test initialization with single recipient."""
        response = RoutingResponse(
            recipients="test@localhost", transform=lambda x: x.upper(), metadata={"priority": "high"}
        )

        assert response.recipients == "test@localhost"
        assert response.transform is not None
        assert response.metadata == {"priority": "high"}

    def test_init_multiple_recipients(self):
        """Test initialization with multiple recipients."""
        recipients = ["user1@localhost", "user2@localhost", "user3@localhost"]

        response = RoutingResponse(
            recipients=recipients,
            transform=lambda x: f"Broadcast: {x}",
            metadata={"type": "broadcast", "timestamp": "2024-01-01"},
        )

        assert response.recipients == recipients
        assert response.transform is not None
        assert response.metadata == {"type": "broadcast", "timestamp": "2024-01-01"}

    def test_transform_function(self):
        """Test transform function functionality."""

        def custom_transform(text):
            return f"[TRANSFORMED] {text.upper()}"

        response = RoutingResponse(recipients="test@localhost", transform=custom_transform)

        result = response.transform("hello world")
        assert result == "[TRANSFORMED] HELLO WORLD"

    def test_transform_lambda(self):
        """Test transform with lambda function."""
        response = RoutingResponse(
            recipients="test@localhost",
            transform=lambda x: x[::-1],  # Reverse string
        )

        result = response.transform("hello")
        assert result == "olleh"

    def test_empty_metadata(self):
        """Test with empty metadata dictionary."""
        response = RoutingResponse(recipients="test@localhost", metadata={})

        assert response.metadata == {}

    def test_none_values(self):
        """Test with None values."""
        response = RoutingResponse(recipients="test@localhost", transform=None, metadata=None)

        assert response.recipients == "test@localhost"
        assert response.transform is None
        assert response.metadata is None

    def test_complex_metadata(self):
        """Test with complex metadata structure."""
        metadata = {
            "priority": "high",
            "category": "urgent",
            "routing_info": {"source": "agent1", "destination": "agent2", "path": ["agent1", "router", "agent2"]},
            "timestamps": [1234567890, 1234567891],
            "flags": {"encrypted": True, "compressed": False},
        }

        response = RoutingResponse(recipients="test@localhost", metadata=metadata)

        assert response.metadata == metadata
        assert response.metadata["routing_info"]["source"] == "agent1"
        assert response.metadata["flags"]["encrypted"] is True

    def test_recipients_list_mutation(self):
        """Test that recipients list can be modified."""
        recipients = ["user1@localhost", "user2@localhost"]
        response = RoutingResponse(recipients=recipients)

        # Modify original list
        recipients.append("user3@localhost")

        # RoutingResponse should reflect the change (same reference)
        assert len(response.recipients) == 3
        assert "user3@localhost" in response.recipients

    def test_dataclass_equality(self):
        """Test dataclass equality comparison."""
        transform_func = lambda x: x.upper()
        metadata = {"key": "value"}

        response1 = RoutingResponse(recipients="test@localhost", transform=transform_func, metadata=metadata)

        response2 = RoutingResponse(recipients="test@localhost", transform=transform_func, metadata=metadata)

        assert response1 == response2

    def test_dataclass_inequality(self):
        """Test dataclass inequality comparison."""
        response1 = RoutingResponse(recipients="test1@localhost")
        response2 = RoutingResponse(recipients="test2@localhost")

        assert response1 != response2

    def test_str_representation(self):
        """Test string representation."""
        response = RoutingResponse(recipients="test@localhost", metadata={"key": "value"})

        str_repr = str(response)
        assert "RoutingResponse" in str_repr
        assert "test@localhost" in str_repr


class TestRoutingFunctionType:
    """Test RoutingFunction type alias."""

    def test_simple_routing_function(self):
        """Test simple routing function that returns string."""

        def simple_router(msg: Message, response: str, context: dict) -> str:
            if "urgent" in response.lower():
                return "urgent@localhost"
            return "normal@localhost"

        # This should be compatible with RoutingFunction type
        routing_func: RoutingFunction = simple_router

        # Test with mock message
        mock_msg = Mock(spec=Message)
        result = routing_func(mock_msg, "This is urgent!", {})
        assert result == "urgent@localhost"

        result = routing_func(mock_msg, "Normal message", {})
        assert result == "normal@localhost"

    def test_routing_function_with_routing_response(self):
        """Test routing function that returns RoutingResponse."""

        def advanced_router(msg: Message, response: str, context: dict) -> RoutingResponse:
            if "broadcast" in response.lower():
                return RoutingResponse(
                    recipients=["user1@localhost", "user2@localhost"],
                    transform=lambda x: f"[BROADCAST] {x}",
                    metadata={"type": "broadcast"},
                )
            return RoutingResponse(recipients="default@localhost")

        # This should be compatible with RoutingFunction type
        routing_func: RoutingFunction = advanced_router

        # Test with mock message
        mock_msg = Mock(spec=Message)
        result = routing_func(mock_msg, "Please broadcast this", {})

        assert isinstance(result, RoutingResponse)
        assert len(result.recipients) == 2
        assert result.transform is not None
        assert result.metadata["type"] == "broadcast"

    def test_routing_function_with_context(self):
        """Test routing function that uses context."""

        def context_router(msg: Message, response: str, context: dict) -> str:
            conversation_id = context.get("conversation_id", "unknown")
            if conversation_id.startswith("priority_"):
                return "priority@localhost"
            return "normal@localhost"

        routing_func: RoutingFunction = context_router

        # Test with different contexts
        mock_msg = Mock(spec=Message)

        result = routing_func(mock_msg, "Test", {"conversation_id": "priority_123"})
        assert result == "priority@localhost"

        result = routing_func(mock_msg, "Test", {"conversation_id": "normal_456"})
        assert result == "normal@localhost"

        result = routing_func(mock_msg, "Test", {})
        assert result == "normal@localhost"

    def test_routing_function_with_message_inspection(self):
        """Test routing function that inspects the original message."""

        def message_inspector(msg: Message, response: str, context: dict) -> str:
            # Check message sender
            if hasattr(msg, "sender") and str(msg.sender).startswith("admin"):
                return "admin@localhost"

            # Check message metadata
            if hasattr(msg, "get_metadata"):
                priority = msg.get_metadata("priority")
                if priority == "high":
                    return "priority@localhost"

            return "default@localhost"

        routing_func: RoutingFunction = message_inspector

        # Test with mock message
        mock_msg = Mock(spec=Message)
        mock_msg.sender = "admin@company.com"
        mock_msg.get_metadata = Mock(return_value=None)

        result = routing_func(mock_msg, "Test", {})
        assert result == "admin@localhost"

    def test_routing_function_complex_logic(self):
        """Test routing function with complex decision logic."""

        def complex_router(msg: Message, response: str, context: dict) -> RoutingResponse:
            # Multi-criteria routing
            recipients = []
            transform = None
            metadata = {"routing_decisions": []}

            # Check response content
            if "error" in response.lower():
                recipients.append("error-handler@localhost")
                metadata["routing_decisions"].append("error_detected")

            if "success" in response.lower():
                recipients.append("success-tracker@localhost")
                metadata["routing_decisions"].append("success_detected")

            # Check context state
            state = context.get("state", {})
            if state.get("interaction_count", 0) > 10:
                recipients.append("long-conversation@localhost")
                metadata["routing_decisions"].append("long_conversation")

            # Default recipient if no specific routing
            if not recipients:
                recipients = ["default@localhost"]
                metadata["routing_decisions"].append("default_routing")

            # Add transformation if multiple recipients
            if len(recipients) > 1:
                transform = lambda x: f"[MULTI-CAST] {x}"

            return RoutingResponse(recipients=recipients, transform=transform, metadata=metadata)

        routing_func: RoutingFunction = complex_router

        # Test various scenarios
        mock_msg = Mock(spec=Message)

        # Error case
        result = routing_func(mock_msg, "An error occurred", {})
        assert "error-handler@localhost" in result.recipients
        assert "error_detected" in result.metadata["routing_decisions"]

        # Success case
        result = routing_func(mock_msg, "Operation successful", {})
        assert "success-tracker@localhost" in result.recipients
        assert "success_detected" in result.metadata["routing_decisions"]

        # Long conversation case
        context = {"state": {"interaction_count": 15}}
        result = routing_func(mock_msg, "Normal message", context)
        assert "long-conversation@localhost" in result.recipients
        assert "long_conversation" in result.metadata["routing_decisions"]

        # Multiple conditions
        result = routing_func(mock_msg, "Success after error", context)
        assert len(result.recipients) == 3  # success, error, long-conversation
        assert result.transform is not None

    def test_routing_function_return_types(self):
        """Test that routing function can return both string and RoutingResponse."""

        def flexible_router(msg: Message, response: str, context: dict):
            if "simple" in response:
                return "simple@localhost"  # String return
            else:
                return RoutingResponse(  # RoutingResponse return
                    recipients="complex@localhost", metadata={"type": "complex"}
                )

        routing_func: RoutingFunction = flexible_router

        mock_msg = Mock(spec=Message)

        # Test string return
        result = routing_func(mock_msg, "simple message", {})
        assert isinstance(result, str)
        assert result == "simple@localhost"

        # Test RoutingResponse return
        result = routing_func(mock_msg, "complex message", {})
        assert isinstance(result, RoutingResponse)
        assert result.recipients == "complex@localhost"
        assert result.metadata["type"] == "complex"


class TestRoutingTypesEdgeCases:
    """Test edge cases for routing types."""

    def test_routing_response_with_empty_recipients_list(self):
        """Test RoutingResponse with empty recipients list."""
        response = RoutingResponse(recipients=[])

        assert response.recipients == []
        assert len(response.recipients) == 0

    def test_routing_response_with_none_recipient(self):
        """Test RoutingResponse with None recipient."""
        # This should be allowed by the type system but might cause issues
        response = RoutingResponse(recipients=None)

        assert response.recipients is None

    def test_transform_function_with_none_input(self):
        """Test transform function handling None input."""

        def safe_transform(text):
            return f"Safe: {text or 'empty'}"

        response = RoutingResponse(recipients="test@localhost", transform=safe_transform)

        result = response.transform(None)
        assert result == "Safe: empty"

    def test_transform_function_exception_handling(self):
        """Test transform function that raises exception."""

        def error_transform(text):
            raise ValueError("Transform error")

        response = RoutingResponse(recipients="test@localhost", transform=error_transform)

        # The exception should propagate
        with pytest.raises(ValueError, match="Transform error"):
            response.transform("test")

    def test_very_large_recipients_list(self):
        """Test with very large recipients list."""
        large_list = [f"user{i}@localhost" for i in range(1000)]

        response = RoutingResponse(recipients=large_list)

        assert len(response.recipients) == 1000
        assert response.recipients[0] == "user0@localhost"
        assert response.recipients[-1] == "user999@localhost"

    def test_deeply_nested_metadata(self):
        """Test with deeply nested metadata structure."""
        nested_metadata = {"level1": {"level2": {"level3": {"level4": {"level5": "deep_value"}}}}}

        response = RoutingResponse(recipients="test@localhost", metadata=nested_metadata)

        assert response.metadata["level1"]["level2"]["level3"]["level4"]["level5"] == "deep_value"

    def test_routing_function_with_no_parameters(self):
        """Test routing function that ignores all parameters."""

        def static_router(msg, response, context):
            return "static@localhost"

        routing_func: RoutingFunction = static_router

        result = routing_func(None, None, None)
        assert result == "static@localhost"

    def test_routing_function_with_side_effects(self):
        """Test routing function that has side effects."""
        call_log = []

        def logging_router(msg: Message, response: str, context: dict) -> str:
            call_log.append({"message": str(msg) if msg else None, "response": response, "context": context})
            return "logged@localhost"

        routing_func: RoutingFunction = logging_router

        mock_msg = Mock(spec=Message)
        mock_msg.__str__ = Mock(return_value="MockMessage")

        result = routing_func(mock_msg, "test response", {"key": "value"})

        assert result == "logged@localhost"
        assert len(call_log) == 1
        assert call_log[0]["response"] == "test response"
        assert call_log[0]["context"]["key"] == "value"
