"""Tests for base guardrail classes and types."""

import pytest

from spade_llm.guardrails.base import Guardrail, GuardrailAction, GuardrailResult


class TestGuardrailAction:
    """Test GuardrailAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        actions = [action.value for action in GuardrailAction]
        expected = ["pass", "modify", "block", "warning"]
        assert set(actions) == set(expected)


class TestGuardrailResult:
    """Test GuardrailResult dataclass."""

    def test_minimal_creation(self):
        """Test creating result with minimal parameters."""
        result = GuardrailResult(action=GuardrailAction.PASS)

        assert result.action == GuardrailAction.PASS
        assert result.content is None
        assert result.reason is None
        assert result.metadata == {}
        assert result.custom_message is None

    def test_full_creation(self):
        """Test creating result with all parameters."""
        metadata = {"key": "value", "count": 42}
        result = GuardrailResult(
            action=GuardrailAction.MODIFY,
            content="modified content",
            reason="Content was altered",
            metadata=metadata,
            custom_message="Custom response",
        )

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "modified content"
        assert result.reason == "Content was altered"
        assert result.metadata == metadata
        assert result.custom_message == "Custom response"

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        result1 = GuardrailResult(action=GuardrailAction.PASS)
        result2 = GuardrailResult(action=GuardrailAction.BLOCK)

        # Ensure separate instances have separate metadata dicts
        result1.metadata["test"] = "value1"
        result2.metadata["test"] = "value2"

        assert result1.metadata["test"] == "value1"
        assert result2.metadata["test"] == "value2"

    def test_equality(self):
        """Test GuardrailResult equality."""
        result1 = GuardrailResult(action=GuardrailAction.PASS, content="content", reason="test")
        result2 = GuardrailResult(action=GuardrailAction.PASS, content="content", reason="test")
        result3 = GuardrailResult(action=GuardrailAction.BLOCK, content="content", reason="test")

        assert result1 == result2
        assert result1 != result3


class ConcreteGuardrail(Guardrail):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, name: str, check_result: GuardrailResult, **kwargs):
        super().__init__(name, **kwargs)
        self.check_result = check_result
        self.check_calls = []

    async def check(self, content: str, context: dict) -> GuardrailResult:
        """Mock implementation that records calls."""
        self.check_calls.append({"content": content, "context": context})
        return self.check_result


class TestGuardrail:
    """Test abstract Guardrail base class."""

    def test_initialization(self):
        """Test guardrail initialization."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteGuardrail(
            name="test_guardrail", check_result=result, enabled=True, blocked_message="Blocked!"
        )

        assert guardrail.name == "test_guardrail"
        assert guardrail.enabled is True
        assert guardrail.blocked_message == "Blocked!"
        assert guardrail._logger.name == "spade_llm.guardrails.test_guardrail"

    def test_initialization_defaults(self):
        """Test guardrail initialization with defaults."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteGuardrail("test", result)

        assert guardrail.enabled is True
        assert guardrail.blocked_message is None

    @pytest.mark.asyncio
    async def test_call_enabled(self):
        """Test calling enabled guardrail."""
        result = GuardrailResult(action=GuardrailAction.MODIFY, content="modified")
        guardrail = ConcreteGuardrail("test", result, enabled=True)

        response = await guardrail("test content", {"key": "value"})

        assert response == result
        assert len(guardrail.check_calls) == 1
        assert guardrail.check_calls[0]["content"] == "test content"
        assert guardrail.check_calls[0]["context"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_call_disabled(self):
        """Test calling disabled guardrail."""
        result = GuardrailResult(action=GuardrailAction.BLOCK)
        guardrail = ConcreteGuardrail("test", result, enabled=False)

        response = await guardrail("test content", {})

        assert response.action == GuardrailAction.PASS
        assert response.content == "test content"
        assert len(guardrail.check_calls) == 0  # Should not call check

    @pytest.mark.asyncio
    async def test_blocked_message_applied(self):
        """Test that blocked_message is applied when blocking."""
        result = GuardrailResult(action=GuardrailAction.BLOCK, reason="blocked")
        guardrail = ConcreteGuardrail("test", result, blocked_message="Custom block message")

        response = await guardrail("content", {})

        assert response.action == GuardrailAction.BLOCK
        assert response.custom_message == "Custom block message"
        assert response.reason == "blocked"  # Original reason preserved

    @pytest.mark.asyncio
    async def test_blocked_message_not_applied_when_not_blocking(self):
        """Test that blocked_message is not applied for non-block actions."""
        result = GuardrailResult(action=GuardrailAction.MODIFY, content="modified")
        guardrail = ConcreteGuardrail("test", result, blocked_message="Should not appear")

        response = await guardrail("content", {})

        assert response.action == GuardrailAction.MODIFY
        assert response.custom_message is None  # Should not be set

    @pytest.mark.asyncio
    async def test_blocked_message_preserves_existing(self):
        """Test that existing custom_message is preserved."""
        result = GuardrailResult(action=GuardrailAction.BLOCK, custom_message="Original message")
        guardrail = ConcreteGuardrail("test", result, blocked_message="New message")

        response = await guardrail("content", {})

        # The original custom_message should be overridden by blocked_message
        assert response.custom_message == "New message"

    @pytest.mark.asyncio
    async def test_multiple_calls(self):
        """Test multiple calls to same guardrail."""
        result = GuardrailResult(action=GuardrailAction.PASS, content="passed")
        guardrail = ConcreteGuardrail("test", result)

        await guardrail("content1", {"ctx": 1})
        await guardrail("content2", {"ctx": 2})

        assert len(guardrail.check_calls) == 2
        assert guardrail.check_calls[0]["content"] == "content1"
        assert guardrail.check_calls[1]["content"] == "content2"
        assert guardrail.check_calls[0]["context"]["ctx"] == 1
        assert guardrail.check_calls[1]["context"]["ctx"] == 2
