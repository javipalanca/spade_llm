"""Tests for guardrail types (Input, Output, Composite)."""

import pytest

from spade_llm.guardrails.base import GuardrailAction, GuardrailResult
from spade_llm.guardrails.types import CompositeGuardrail, InputGuardrail, OutputGuardrail


class ConcreteInputGuardrail(InputGuardrail):
    """Concrete implementation for testing InputGuardrail."""

    def __init__(self, name: str, result: GuardrailResult, **kwargs):
        super().__init__(name, **kwargs)
        self.result = result
        self.call_log = []

    async def check(self, content: str, context: dict) -> GuardrailResult:
        self.call_log.append({"content": content, "context": context})
        return self.result


class ConcreteOutputGuardrail(OutputGuardrail):
    """Concrete implementation for testing OutputGuardrail."""

    def __init__(self, name: str, result: GuardrailResult, **kwargs):
        super().__init__(name, **kwargs)
        self.result = result
        self.call_log = []

    async def check(self, content: str, context: dict) -> GuardrailResult:
        self.call_log.append({"content": content, "context": context})
        return self.result


class TestInputGuardrail:
    """Test InputGuardrail class."""

    def test_initialization_defaults(self):
        """Test InputGuardrail initialization with defaults."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteInputGuardrail("input_test", result)

        assert guardrail.name == "input_test"
        assert guardrail.enabled is True
        assert guardrail.blocked_message == "Your message was blocked by security filters."

    def test_initialization_custom_message(self):
        """Test InputGuardrail with custom blocked message."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteInputGuardrail("input_test", result, blocked_message="Custom input block message")

        assert guardrail.blocked_message == "Custom input block message"

    def test_initialization_other_params(self):
        """Test InputGuardrail with other parameters."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteInputGuardrail("input_test", result, enabled=False)

        assert guardrail.enabled is False

    @pytest.mark.asyncio
    async def test_check_method_called(self):
        """Test that check method is properly called."""
        result = GuardrailResult(action=GuardrailAction.PASS, content="passed")
        guardrail = ConcreteInputGuardrail("input_test", result)

        response = await guardrail("test input", {"key": "value"})

        assert response == result
        assert len(guardrail.call_log) == 1
        assert guardrail.call_log[0]["content"] == "test input"
        assert guardrail.call_log[0]["context"] == {"key": "value"}


class TestOutputGuardrail:
    """Test OutputGuardrail class."""

    def test_initialization_defaults(self):
        """Test OutputGuardrail initialization with defaults."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteOutputGuardrail("output_test", result)

        assert guardrail.name == "output_test"
        assert guardrail.enabled is True
        assert guardrail.blocked_message == "I apologize, but I cannot provide that response."

    def test_initialization_custom_message(self):
        """Test OutputGuardrail with custom blocked message."""
        result = GuardrailResult(action=GuardrailAction.PASS)
        guardrail = ConcreteOutputGuardrail("output_test", result, blocked_message="Custom output block message")

        assert guardrail.blocked_message == "Custom output block message"

    @pytest.mark.asyncio
    async def test_check_method_called(self):
        """Test that check method is properly called."""
        result = GuardrailResult(action=GuardrailAction.MODIFY, content="modified")
        guardrail = ConcreteOutputGuardrail("output_test", result)

        response = await guardrail("test output", {"context": "data"})

        assert response == result
        assert len(guardrail.call_log) == 1
        assert guardrail.call_log[0]["content"] == "test output"
        assert guardrail.call_log[0]["context"] == {"context": "data"}


class TestCompositeGuardrail:
    """Test CompositeGuardrail class."""

    def test_initialization(self):
        """Test CompositeGuardrail initialization."""
        child1 = ConcreteInputGuardrail("child1", GuardrailResult(action=GuardrailAction.PASS))
        child2 = ConcreteInputGuardrail("child2", GuardrailResult(action=GuardrailAction.PASS))

        composite = CompositeGuardrail(
            name="composite_test", guardrails=[child1, child2], stop_on_block=True, blocked_message="Composite blocked"
        )

        assert composite.name == "composite_test"
        assert composite.guardrails == [child1, child2]
        assert composite.stop_on_block is True
        assert composite.blocked_message == "Composite blocked"

    def test_initialization_defaults(self):
        """Test CompositeGuardrail with defaults."""
        child = ConcreteInputGuardrail("child", GuardrailResult(action=GuardrailAction.PASS))
        composite = CompositeGuardrail("test", [child])

        assert composite.stop_on_block is True
        assert composite.enabled is True
        assert composite.blocked_message is None

    @pytest.mark.asyncio
    async def test_all_pass(self):
        """Test composite when all guardrails pass."""
        child1 = ConcreteInputGuardrail("child1", GuardrailResult(action=GuardrailAction.PASS, content="content"))
        child2 = ConcreteInputGuardrail("child2", GuardrailResult(action=GuardrailAction.PASS, content="content"))

        composite = CompositeGuardrail("test", [child1, child2])

        result = await composite.check("test content", {"key": "value"})

        assert result.action == GuardrailAction.PASS
        assert result.content == "test content"
        assert len(child1.call_log) == 1
        assert len(child2.call_log) == 1

    @pytest.mark.asyncio
    async def test_first_blocks_stop_on_block(self):
        """Test composite when first guardrail blocks and stop_on_block=True."""
        child1 = ConcreteInputGuardrail("child1", GuardrailResult(action=GuardrailAction.BLOCK, reason="blocked"))
        child2 = ConcreteInputGuardrail("child2", GuardrailResult(action=GuardrailAction.PASS))

        composite = CompositeGuardrail("test", [child1, child2], stop_on_block=True)

        result = await composite.check("test content", {})

        assert result.action == GuardrailAction.BLOCK
        assert result.reason == "blocked"
        assert len(child1.call_log) == 1
        assert len(child2.call_log) == 0  # Should not be called

    @pytest.mark.asyncio
    async def test_first_blocks_no_stop_on_block(self):
        """Test composite when first guardrail blocks and stop_on_block=False."""
        child1 = ConcreteInputGuardrail("child1", GuardrailResult(action=GuardrailAction.BLOCK, reason="block1"))
        child2 = ConcreteInputGuardrail("child2", GuardrailResult(action=GuardrailAction.PASS))

        composite = CompositeGuardrail("test", [child1, child2], stop_on_block=False)

        result = await composite.check("test content", {})

        assert result.action == GuardrailAction.PASS  # Continue processing
        assert len(child1.call_log) == 1
        assert len(child2.call_log) == 1  # Should be called

    @pytest.mark.asyncio
    async def test_modify_chain(self):
        """Test composite with multiple modifications."""
        child1 = ConcreteInputGuardrail(
            "child1", GuardrailResult(action=GuardrailAction.MODIFY, content="modified1", reason="first mod")
        )
        child2 = ConcreteInputGuardrail(
            "child2", GuardrailResult(action=GuardrailAction.MODIFY, content="modified2", reason="second mod")
        )

        composite = CompositeGuardrail("test", [child1, child2])

        result = await composite.check("original", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "modified2"  # Last modification wins
        assert "child1: first mod; child2: second mod" in result.reason

        # Verify content flows through
        assert child1.call_log[0]["content"] == "original"
        assert child2.call_log[0]["content"] == "modified1"

    @pytest.mark.asyncio
    async def test_warning_accumulation(self):
        """Test composite with warnings."""
        child1 = ConcreteInputGuardrail("child1", GuardrailResult(action=GuardrailAction.WARNING, reason="warning1"))
        child2 = ConcreteInputGuardrail("child2", GuardrailResult(action=GuardrailAction.PASS))

        composite = CompositeGuardrail("test", [child1, child2])

        result = await composite.check("content", {})

        assert result.action == GuardrailAction.PASS
        assert result.content == "content"
        # Warnings should not affect final result but may be logged in reason

    @pytest.mark.asyncio
    async def test_disabled_child_skipped(self):
        """Test that disabled child guardrails are skipped."""
        child1 = ConcreteInputGuardrail("child1", GuardrailResult(action=GuardrailAction.PASS))
        child1.enabled = False
        child2 = ConcreteInputGuardrail("child2", GuardrailResult(action=GuardrailAction.PASS))

        composite = CompositeGuardrail("test", [child1, child2])

        result = await composite.check("content", {})

        assert result.action == GuardrailAction.PASS
        assert len(child1.call_log) == 0  # Should be skipped
        assert len(child2.call_log) == 1  # Should be called

    @pytest.mark.asyncio
    async def test_empty_guardrails_list(self):
        """Test composite with empty guardrails list."""
        composite = CompositeGuardrail("test", [])

        result = await composite.check("content", {})

        assert result.action == GuardrailAction.PASS
        assert result.content == "content"

    @pytest.mark.asyncio
    async def test_composite_blocked_message_override(self):
        """Test that composite blocked_message overrides child message."""
        child = ConcreteInputGuardrail(
            "child", GuardrailResult(action=GuardrailAction.BLOCK, custom_message="Child message")
        )

        composite = CompositeGuardrail("test", [child], blocked_message="Composite message")

        result = await composite.check("content", {})

        assert result.action == GuardrailAction.BLOCK
        assert result.custom_message == "Composite message"

    @pytest.mark.asyncio
    async def test_complex_scenario(self):
        """Test complex scenario with multiple actions."""
        # First modifies, second warns, third passes
        child1 = ConcreteInputGuardrail(
            "modifier", GuardrailResult(action=GuardrailAction.MODIFY, content="step1", reason="modified content")
        )
        child2 = ConcreteInputGuardrail(
            "warner", GuardrailResult(action=GuardrailAction.WARNING, reason="suspicious content")
        )
        child3 = ConcreteInputGuardrail("passer", GuardrailResult(action=GuardrailAction.PASS))

        composite = CompositeGuardrail("complex", [child1, child2, child3])

        result = await composite.check("original", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "step1"
        assert "modifier: modified content" in result.reason
        assert "warner: suspicious content" in result.reason

        # Verify call chain
        assert child1.call_log[0]["content"] == "original"
        assert child2.call_log[0]["content"] == "step1"
        assert child3.call_log[0]["content"] == "step1"
