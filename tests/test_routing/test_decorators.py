"""Tests for routing decorators."""

from unittest.mock import Mock

import pytest
from spade.message import Message

from spade_llm.routing.decorators import routing_rule


class TestRoutingRuleDecorator:
    """Test routing_rule decorator."""

    def test_decorator_without_parameters(self):
        """Test decorator applied without parameters."""

        @routing_rule()
        def simple_rule(msg, response, context):
            return "test@localhost"

        # Check that function works normally
        mock_msg = Mock(spec=Message)
        result = simple_rule(mock_msg, "test", {})
        assert result == "test@localhost"

        # Check that decorator metadata was added
        assert hasattr(simple_rule, "_is_routing_rule")
        assert simple_rule._is_routing_rule is True
        assert hasattr(simple_rule, "_priority")
        assert simple_rule._priority == 0
        assert hasattr(simple_rule, "_rule_name")
        assert simple_rule._rule_name == "simple_rule"

    def test_decorator_with_priority(self):
        """Test decorator with priority parameter."""

        @routing_rule(priority=10)
        def priority_rule(msg, response, context):
            if "urgent" in response.lower():
                return "urgent@localhost"
            return None

        # Check function works
        mock_msg = Mock(spec=Message)
        result = priority_rule(mock_msg, "urgent message", {})
        assert result == "urgent@localhost"

        result = priority_rule(mock_msg, "normal message", {})
        assert result is None

        # Check metadata
        assert priority_rule._is_routing_rule is True
        assert priority_rule._priority == 10
        assert priority_rule._rule_name == "priority_rule"

    def test_decorator_with_name(self):
        """Test decorator with custom name parameter."""

        @routing_rule(name="custom_routing_rule")
        def some_function(msg, response, context):
            return "named@localhost"

        # Check function works
        mock_msg = Mock(spec=Message)
        result = some_function(mock_msg, "test", {})
        assert result == "named@localhost"

        # Check metadata
        assert some_function._is_routing_rule is True
        assert some_function._priority == 0
        assert some_function._rule_name == "custom_routing_rule"

    def test_decorator_with_priority_and_name(self):
        """Test decorator with both priority and name parameters."""

        @routing_rule(priority=5, name="high_priority_rule")
        def complex_rule(msg, response, context):
            conversation_id = context.get("conversation_id", "")
            if conversation_id.startswith("admin_"):
                return "admin@localhost"
            return "user@localhost"

        # Check function works
        mock_msg = Mock(spec=Message)
        context1 = {"conversation_id": "admin_123"}
        result = complex_rule(mock_msg, "test", context1)
        assert result == "admin@localhost"

        context2 = {"conversation_id": "user_456"}
        result = complex_rule(mock_msg, "test", context2)
        assert result == "user@localhost"

        # Check metadata
        assert complex_rule._is_routing_rule is True
        assert complex_rule._priority == 5
        assert complex_rule._rule_name == "high_priority_rule"

    def test_decorator_preserves_function_properties(self):
        """Test that decorator preserves original function properties."""

        @routing_rule(priority=3, name="documented_rule")
        def documented_function(msg, response, context):
            """This is a documented routing rule function."""
            return "documented@localhost"

        # Check that docstring is preserved
        assert documented_function.__doc__ == "This is a documented routing rule function."

        # Check that function name is preserved (functools.wraps)
        assert documented_function.__name__ == "documented_function"

        # Check that the function is callable
        assert callable(documented_function)

        # Check metadata
        assert documented_function._is_routing_rule is True
        assert documented_function._priority == 3
        assert documented_function._rule_name == "documented_rule"

    def test_decorator_with_negative_priority(self):
        """Test decorator with negative priority."""

        @routing_rule(priority=-5)
        def low_priority_rule(msg, response, context):
            return "low@localhost"

        mock_msg = Mock(spec=Message)
        result = low_priority_rule(mock_msg, "test", {})
        assert result == "low@localhost"

        assert low_priority_rule._priority == -5

    def test_decorator_with_zero_priority(self):
        """Test decorator with explicit zero priority."""

        @routing_rule(priority=0)
        def zero_priority_rule(msg, response, context):
            return "zero@localhost"

        mock_msg = Mock(spec=Message)
        result = zero_priority_rule(mock_msg, "test", {})
        assert result == "zero@localhost"

        assert zero_priority_rule._priority == 0

    def test_decorator_with_none_name(self):
        """Test decorator with None name (should use function name)."""

        @routing_rule(name=None)
        def none_name_rule(msg, response, context):
            return "none@localhost"

        mock_msg = Mock(spec=Message)
        result = none_name_rule(mock_msg, "test", {})
        assert result == "none@localhost"

        # Should default to function name when name is None
        assert none_name_rule._rule_name == "none_name_rule"

    def test_multiple_decorated_functions(self):
        """Test multiple functions with different decorator parameters."""

        @routing_rule(priority=10, name="first_rule")
        def first_rule(msg, response, context):
            return "first@localhost"

        @routing_rule(priority=5, name="second_rule")
        def second_rule(msg, response, context):
            return "second@localhost"

        @routing_rule(priority=1)
        def third_rule(msg, response, context):
            return "third@localhost"

        # Check all functions work independently
        mock_msg = Mock(spec=Message)

        assert first_rule(mock_msg, "test", {}) == "first@localhost"
        assert second_rule(mock_msg, "test", {}) == "second@localhost"
        assert third_rule(mock_msg, "test", {}) == "third@localhost"

        # Check metadata is different for each
        assert first_rule._priority == 10
        assert second_rule._priority == 5
        assert third_rule._priority == 1

        assert first_rule._rule_name == "first_rule"
        assert second_rule._rule_name == "second_rule"
        assert third_rule._rule_name == "third_rule"

    def test_decorated_function_with_complex_logic(self):
        """Test decorator on function with complex routing logic."""

        @routing_rule(priority=8, name="complex_router")
        def complex_routing_rule(msg, response, context):
            # Multi-condition routing
            if "error" in response.lower() and "critical" in response.lower():
                return "critical-error@localhost"
            elif "error" in response.lower():
                return "error@localhost"
            elif "success" in response.lower():
                return "success@localhost"
            elif context.get("conversation_id", "").startswith("admin_"):
                return "admin@localhost"
            else:
                return "default@localhost"

        mock_msg = Mock(spec=Message)

        # Test various conditions
        result = complex_routing_rule(mock_msg, "Critical error occurred", {})
        assert result == "critical-error@localhost"

        result = complex_routing_rule(mock_msg, "An error happened", {})
        assert result == "error@localhost"

        result = complex_routing_rule(mock_msg, "Success message", {})
        assert result == "success@localhost"

        result = complex_routing_rule(mock_msg, "Normal message", {"conversation_id": "admin_123"})
        assert result == "admin@localhost"

        result = complex_routing_rule(mock_msg, "Normal message", {"conversation_id": "user_456"})
        assert result == "default@localhost"

        # Check metadata
        assert complex_routing_rule._is_routing_rule is True
        assert complex_routing_rule._priority == 8
        assert complex_routing_rule._rule_name == "complex_router"

    def test_decorated_function_returning_routing_response(self):
        """Test decorator on function that returns RoutingResponse."""
        from spade_llm.routing.types import RoutingResponse

        @routing_rule(priority=7, name="response_rule")
        def routing_response_rule(msg, response, context):
            if "broadcast" in response.lower():
                return RoutingResponse(
                    recipients=["user1@localhost", "user2@localhost"],
                    transform=lambda x: f"[BROADCAST] {x}",
                    metadata={"type": "broadcast"},
                )
            return RoutingResponse(recipients="single@localhost")

        mock_msg = Mock(spec=Message)

        # Test broadcast case
        result = routing_response_rule(mock_msg, "Please broadcast this", {})
        assert isinstance(result, RoutingResponse)
        assert len(result.recipients) == 2
        assert result.transform is not None
        assert result.metadata["type"] == "broadcast"

        # Test single recipient case
        result = routing_response_rule(mock_msg, "Normal message", {})
        assert isinstance(result, RoutingResponse)
        assert result.recipients == "single@localhost"

        # Check metadata
        assert routing_response_rule._is_routing_rule is True
        assert routing_response_rule._priority == 7
        assert routing_response_rule._rule_name == "response_rule"

    def test_decorated_function_with_exception(self):
        """Test decorator on function that raises exception."""

        @routing_rule(priority=2, name="error_rule")
        def error_rule(msg, response, context):
            if "crash" in response.lower():
                raise ValueError("Intentional crash")
            return "safe@localhost"

        mock_msg = Mock(spec=Message)

        # Normal case should work
        result = error_rule(mock_msg, "safe message", {})
        assert result == "safe@localhost"

        # Exception case should still raise
        with pytest.raises(ValueError, match="Intentional crash"):
            error_rule(mock_msg, "crash now", {})

        # Check metadata
        assert error_rule._is_routing_rule is True
        assert error_rule._priority == 2
        assert error_rule._rule_name == "error_rule"

    def test_decorator_preserves_function_signature(self):
        """Test that decorator preserves function signature."""

        @routing_rule(priority=1)
        def signature_test(msg, response, context, extra_param=None):
            if extra_param:
                return f"{extra_param}@localhost"
            return "default@localhost"

        mock_msg = Mock(spec=Message)

        # Test with default parameter
        result = signature_test(mock_msg, "test", {})
        assert result == "default@localhost"

        # Test with extra parameter
        result = signature_test(mock_msg, "test", {}, extra_param="custom")
        assert result == "custom@localhost"

        # Check metadata
        assert signature_test._is_routing_rule is True

    def test_decorator_priority_sorting(self):
        """Test that priority can be used for sorting decorated functions."""

        @routing_rule(priority=1)
        def low_priority():
            pass

        @routing_rule(priority=10)
        def high_priority():
            pass

        @routing_rule(priority=5)
        def medium_priority():
            pass

        # Create list of functions
        functions = [low_priority, high_priority, medium_priority]

        # Sort by priority (higher first)
        sorted_functions = sorted(functions, key=lambda f: f._priority, reverse=True)

        assert sorted_functions[0] is high_priority
        assert sorted_functions[1] is medium_priority
        assert sorted_functions[2] is low_priority

    def test_decorator_rule_identification(self):
        """Test that decorated functions can be identified as routing rules."""

        @routing_rule()
        def routing_function():
            pass

        def normal_function():
            pass

        # Test identification
        assert hasattr(routing_function, "_is_routing_rule")
        assert routing_function._is_routing_rule is True

        assert not hasattr(normal_function, "_is_routing_rule")

        # Test filtering
        all_functions = [routing_function, normal_function]
        routing_rules = [f for f in all_functions if getattr(f, "_is_routing_rule", False)]

        assert len(routing_rules) == 1
        assert routing_rules[0] is routing_function


class TestRoutingRuleDecoratorEdgeCases:
    """Test edge cases for routing_rule decorator."""

    def test_decorator_on_lambda(self):
        """Test decorator applied to lambda function."""
        # Note: This is not recommended usage but should work
        decorated_lambda = routing_rule(priority=3)(lambda msg, response, context: "lambda@localhost")

        mock_msg = Mock(spec=Message)
        result = decorated_lambda(mock_msg, "test", {})
        assert result == "lambda@localhost"

        assert decorated_lambda._is_routing_rule is True
        assert decorated_lambda._priority == 3
        assert decorated_lambda._rule_name == "<lambda>"

    def test_decorator_with_very_high_priority(self):
        """Test decorator with very high priority value."""

        @routing_rule(priority=999999)
        def extreme_priority(msg, response, context):
            return "extreme@localhost"

        assert extreme_priority._priority == 999999

    def test_decorator_with_very_long_name(self):
        """Test decorator with very long name."""
        long_name = "a" * 1000

        @routing_rule(name=long_name)
        def long_name_rule(msg, response, context):
            return "long@localhost"

        assert long_name_rule._rule_name == long_name
        assert len(long_name_rule._rule_name) == 1000

    def test_decorator_with_special_characters_in_name(self):
        """Test decorator with special characters in name."""
        special_name = "rule-with_special.chars@#$%"

        @routing_rule(name=special_name)
        def special_char_rule(msg, response, context):
            return "special@localhost"

        assert special_char_rule._rule_name == special_name

    def test_decorator_metadata_immutability(self):
        """Test that decorator metadata can't be easily modified."""

        @routing_rule(priority=5, name="protected_rule")
        def protected_rule(msg, response, context):
            return "protected@localhost"

        # Try to modify (this should work but would be unexpected behavior)
        protected_rule._priority = 999
        protected_rule._rule_name = "modified"
        protected_rule._is_routing_rule = False

        # Check that changes took effect (Python doesn't prevent this)
        assert protected_rule._priority == 999
        assert protected_rule._rule_name == "modified"
        assert protected_rule._is_routing_rule is False

        # Function should still work
        mock_msg = Mock(spec=Message)
        result = protected_rule(mock_msg, "test", {})
        assert result == "protected@localhost"

    def test_decorator_on_method(self):
        """Test decorator applied to class method."""

        class RouterClass:
            @routing_rule(priority=4, name="method_rule")
            def routing_method(self, msg, response, context):
                return "method@localhost"

        router = RouterClass()

        # Check that method works
        mock_msg = Mock(spec=Message)
        result = router.routing_method(mock_msg, "test", {})
        assert result == "method@localhost"

        # Check metadata (should be on the unbound method)
        assert hasattr(router.routing_method, "_is_routing_rule")
        assert router.routing_method._is_routing_rule is True
        assert router.routing_method._priority == 4
        assert router.routing_method._rule_name == "method_rule"

    def test_decorator_on_static_method(self):
        """Test decorator applied to static method."""

        class RouterClass:
            @staticmethod
            @routing_rule(priority=6, name="static_rule")
            def static_routing_method(msg, response, context):
                return "static@localhost"

        # Check that static method works
        mock_msg = Mock(spec=Message)
        result = RouterClass.static_routing_method(mock_msg, "test", {})
        assert result == "static@localhost"

        # Check metadata
        assert RouterClass.static_routing_method._is_routing_rule is True
        assert RouterClass.static_routing_method._priority == 6
        assert RouterClass.static_routing_method._rule_name == "static_rule"

    def test_multiple_decorator_applications(self):
        """Test applying decorator multiple times (not recommended but possible)."""

        @routing_rule(priority=1, name="first")
        @routing_rule(priority=2, name="second")
        def double_decorated(msg, response, context):
            return "double@localhost"

        # The outer decorator should take precedence
        assert double_decorated._priority == 1
        assert double_decorated._rule_name == "first"
        assert double_decorated._is_routing_rule is True

        # Function should still work
        mock_msg = Mock(spec=Message)
        result = double_decorated(mock_msg, "test", {})
        assert result == "double@localhost"
