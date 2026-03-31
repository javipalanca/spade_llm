"""Tests for concrete guardrail implementations."""

import pytest

from spade_llm.guardrails.base import GuardrailAction, GuardrailResult
from spade_llm.guardrails.implementations import CustomFunctionGuardrail, KeywordGuardrail, RegexGuardrail


class TestKeywordGuardrail:
    """Test KeywordGuardrail implementation."""

    def test_initialization(self):
        """Test KeywordGuardrail initialization."""
        guardrail = KeywordGuardrail(
            name="keyword_test",
            blocked_keywords=["bad", "harmful"],
            action=GuardrailAction.BLOCK,
            replacement="[CENSORED]",
            case_sensitive=True,
            blocked_message="Blocked by keyword filter",
        )

        assert guardrail.name == "keyword_test"
        assert guardrail.blocked_keywords == ["bad", "harmful"]
        assert guardrail.action == GuardrailAction.BLOCK
        assert guardrail.replacement == "[CENSORED]"
        assert guardrail.case_sensitive is True
        assert guardrail.blocked_message == "Blocked by keyword filter"

    def test_initialization_case_insensitive(self):
        """Test case insensitive initialization."""
        guardrail = KeywordGuardrail(name="test", blocked_keywords=["BAD", "Harmful"], case_sensitive=False)

        assert guardrail.blocked_keywords == ["bad", "harmful"]

    def test_initialization_defaults(self):
        """Test default values."""
        guardrail = KeywordGuardrail(name="test", blocked_keywords=["test"])

        assert guardrail.action == GuardrailAction.BLOCK
        assert guardrail.replacement == "[REDACTED]"
        assert guardrail.case_sensitive is False

    @pytest.mark.asyncio
    async def test_block_action_case_sensitive(self):
        """Test blocking with case sensitive matching."""
        guardrail = KeywordGuardrail(
            name="test", blocked_keywords=["BadWord"], action=GuardrailAction.BLOCK, case_sensitive=True
        )

        # Should block exact case
        result = await guardrail.check("This contains BadWord", {})
        assert result.action == GuardrailAction.BLOCK
        assert "BadWord" in result.reason

        # Should not block different case
        result = await guardrail.check("This contains badword", {})
        assert result.action == GuardrailAction.PASS

    @pytest.mark.asyncio
    async def test_block_action_case_insensitive(self):
        """Test blocking with case insensitive matching."""
        guardrail = KeywordGuardrail(
            name="test", blocked_keywords=["BadWord"], action=GuardrailAction.BLOCK, case_sensitive=False
        )

        # Should block any case variation
        test_cases = ["BadWord", "badword", "BADWORD", "bAdWoRd"]

        for test_case in test_cases:
            result = await guardrail.check(f"Text with {test_case}", {})
            assert result.action == GuardrailAction.BLOCK, f"Failed for: {test_case}"

    @pytest.mark.asyncio
    async def test_modify_action_case_sensitive(self):
        """Test modification with case sensitive matching."""
        guardrail = KeywordGuardrail(
            name="test",
            blocked_keywords=["replace"],
            action=GuardrailAction.MODIFY,
            replacement="[FIXED]",
            case_sensitive=True,
        )

        result = await guardrail.check("Please replace this", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "Please [FIXED] this"
        assert "replace" in result.reason

    @pytest.mark.asyncio
    async def test_modify_action_case_insensitive(self):
        """Test modification with case insensitive matching."""
        guardrail = KeywordGuardrail(
            name="test",
            blocked_keywords=["replace"],
            action=GuardrailAction.MODIFY,
            replacement="[FIXED]",
            case_sensitive=False,
        )

        result = await guardrail.check("Please REPLACE this Replace", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "Please [FIXED] this [FIXED]"

    @pytest.mark.asyncio
    async def test_multiple_keywords(self):
        """Test with multiple blocked keywords."""
        guardrail = KeywordGuardrail(
            name="test", blocked_keywords=["bad", "harmful", "inappropriate"], action=GuardrailAction.BLOCK
        )

        # Test each keyword
        for keyword in ["bad", "harmful", "inappropriate"]:
            result = await guardrail.check(f"This is {keyword} content", {})
            assert result.action == GuardrailAction.BLOCK
            assert keyword in result.reason

    @pytest.mark.asyncio
    async def test_no_match_passes(self):
        """Test that clean content passes through."""
        guardrail = KeywordGuardrail(name="test", blocked_keywords=["bad", "harmful"], action=GuardrailAction.BLOCK)

        result = await guardrail.check("This is clean content", {})

        assert result.action == GuardrailAction.PASS
        assert result.content == "This is clean content"

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test with empty content."""
        guardrail = KeywordGuardrail(name="test", blocked_keywords=["bad"], action=GuardrailAction.BLOCK)

        result = await guardrail.check("", {})

        assert result.action == GuardrailAction.PASS
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_keyword_as_substring(self):
        """Test that keywords work as substrings."""
        guardrail = KeywordGuardrail(name="test", blocked_keywords=["bad"], action=GuardrailAction.BLOCK)

        result = await guardrail.check("This is really badly written", {})

        assert result.action == GuardrailAction.BLOCK
        assert "bad" in result.reason


class TestRegexGuardrail:
    """Test RegexGuardrail implementation."""

    def test_initialization(self):
        """Test RegexGuardrail initialization."""
        patterns = {r"\d{3}-\d{2}-\d{4}": "[SSN]", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL]"}

        guardrail = RegexGuardrail(name="regex_test", patterns=patterns, blocked_message="Pattern detected")

        assert guardrail.name == "regex_test"
        assert guardrail.patterns == patterns
        assert guardrail.blocked_message == "Pattern detected"

    @pytest.mark.asyncio
    async def test_replacement_patterns(self):
        """Test regex patterns that replace text."""
        patterns = {r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL]", r"\d{3}-\d{2}-\d{4}": "[SSN]"}

        guardrail = RegexGuardrail("test", patterns)

        result = await guardrail.check("Contact john@example.com or call 123-45-6789", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "Contact [EMAIL] or call [SSN]"
        assert "Patterns replaced" in result.reason

    @pytest.mark.asyncio
    async def test_blocking_patterns(self):
        """Test regex patterns that block content."""
        patterns = {
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b": GuardrailAction.BLOCK  # Credit card pattern
        }

        guardrail = RegexGuardrail("test", patterns, blocked_message="Sensitive data detected")

        result = await guardrail.check("My card number is 1234 5678 9012 3456", {})

        assert result.action == GuardrailAction.BLOCK
        assert result.custom_message == "Sensitive data detected"
        assert "Pattern" in result.reason

    @pytest.mark.asyncio
    async def test_mixed_patterns(self):
        """Test combination of replacement and blocking patterns."""
        patterns = {
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL]",
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b": GuardrailAction.BLOCK,
        }

        guardrail = RegexGuardrail("test", patterns)

        # Should replace email
        result = await guardrail.check("Email me at test@example.com", {})
        assert result.action == GuardrailAction.MODIFY
        assert result.content == "Email me at [EMAIL]"

        # Should block credit card
        result = await guardrail.check("Card: 1234 5678 9012 3456", {})
        assert result.action == GuardrailAction.BLOCK

    @pytest.mark.asyncio
    async def test_no_match_passes(self):
        """Test that content without matches passes through."""
        patterns = {r"\d{3}-\d{2}-\d{4}": "[SSN]"}

        guardrail = RegexGuardrail("test", patterns)

        result = await guardrail.check("This is clean text without patterns", {})

        assert result.action == GuardrailAction.PASS
        assert result.content == "This is clean text without patterns"

    @pytest.mark.asyncio
    async def test_multiple_replacements_same_pattern(self):
        """Test multiple instances of same pattern."""
        patterns = {r"\b\d{3}-\d{2}-\d{4}\b": "[SSN]"}

        guardrail = RegexGuardrail("test", patterns)

        result = await guardrail.check("SSNs: 123-45-6789 and 987-65-4321", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "SSNs: [SSN] and [SSN]"

    @pytest.mark.asyncio
    async def test_empty_patterns(self):
        """Test with empty patterns dictionary."""
        guardrail = RegexGuardrail("test", {})

        result = await guardrail.check("Any content", {})

        assert result.action == GuardrailAction.PASS
        assert result.content == "Any content"

    @pytest.mark.asyncio
    async def test_invalid_regex_handled_gracefully(self):
        """Test that invalid regex patterns are handled gracefully."""
        # This would be caught during compilation, but let's test the general flow
        patterns = {
            r"[a-z]+": "[LETTERS]"  # Valid pattern
        }

        guardrail = RegexGuardrail("test", patterns)

        result = await guardrail.check("test123", {})

        assert result.action == GuardrailAction.MODIFY
        assert result.content == "[LETTERS]123"


class TestCustomFunctionGuardrail:
    """Test CustomFunctionGuardrail implementation."""

    def test_initialization(self, simple_custom_function):
        """Test CustomFunctionGuardrail initialization."""
        guardrail = CustomFunctionGuardrail(
            name="custom_test", check_function=simple_custom_function, blocked_message="Custom function blocked"
        )

        assert guardrail.name == "custom_test"
        assert guardrail.check_function == simple_custom_function
        assert guardrail.blocked_message == "Custom function blocked"

    @pytest.mark.asyncio
    async def test_sync_function_execution(self, simple_custom_function):
        """Test execution of synchronous custom function."""
        guardrail = CustomFunctionGuardrail("test", simple_custom_function)

        # Test content under limit (should pass)
        result = await guardrail.check("Short content", {"key": "value"})
        assert result.action == GuardrailAction.PASS

        # Test content over limit (should block)
        long_content = "x" * 101
        result = await guardrail.check(long_content, {})
        assert result.action == GuardrailAction.BLOCK
        assert "too long" in result.reason

    @pytest.mark.asyncio
    async def test_modify_function(self, modify_custom_function):
        """Test custom function that modifies content."""
        guardrail = CustomFunctionGuardrail("test", modify_custom_function)

        # Content without prefix should be modified
        result = await guardrail.check("Original content", {})
        assert result.action == GuardrailAction.MODIFY
        assert result.content == "[CHECKED] Original content"
        assert "safety prefix" in result.reason

        # Content with prefix should pass
        result = await guardrail.check("[CHECKED] Already prefixed", {})
        assert result.action == GuardrailAction.PASS

    @pytest.mark.asyncio
    async def test_blocked_message_applied(self, simple_custom_function):
        """Test that custom blocked message is applied."""
        guardrail = CustomFunctionGuardrail("test", simple_custom_function, blocked_message="Custom block message")

        long_content = "x" * 101
        result = await guardrail.check(long_content, {})

        assert result.action == GuardrailAction.BLOCK
        assert result.custom_message == "Custom block message"

    @pytest.mark.asyncio
    async def test_blocked_message_preserves_existing(self):
        """Test that existing custom_message in result is preserved."""

        def custom_with_message(content: str, context: dict) -> GuardrailResult:
            return GuardrailResult(action=GuardrailAction.BLOCK, custom_message="Function's own message")

        guardrail = CustomFunctionGuardrail("test", custom_with_message, blocked_message="Guardrail message")

        result = await guardrail.check("content", {})

        # The function's message should be preserved (not overridden)
        assert result.custom_message == "Function's own message"

    @pytest.mark.asyncio
    async def test_function_receives_context(self):
        """Test that custom function receives context correctly."""
        received_context = {}

        def context_recorder(content: str, context: dict) -> GuardrailResult:
            received_context.update(context)
            return GuardrailResult(action=GuardrailAction.PASS, content=content)

        guardrail = CustomFunctionGuardrail("test", context_recorder)

        test_context = {"key": "value", "number": 42}
        await guardrail.check("test content", test_context)

        assert received_context == test_context

    @pytest.mark.asyncio
    async def test_function_error_propagated(self):
        """Test that errors in custom function are propagated."""

        def error_function(content: str, context: dict) -> GuardrailResult:
            raise ValueError("Custom function error")

        guardrail = CustomFunctionGuardrail("test", error_function)

        with pytest.raises(ValueError, match="Custom function error"):
            await guardrail.check("content", {})

    @pytest.mark.asyncio
    async def test_complex_custom_function(self):
        """Test complex custom function with multiple conditions."""

        def complex_checker(content: str, context: dict) -> GuardrailResult:
            if "urgent" in content.lower():
                if context.get("business_hours", True):
                    return GuardrailResult(action=GuardrailAction.PASS, content=content)
                else:
                    return GuardrailResult(
                        action=GuardrailAction.MODIFY,
                        content=content + " [Note: Received outside business hours]",
                        reason="Added business hours notice",
                    )
            elif len(content) > 50:
                return GuardrailResult(action=GuardrailAction.BLOCK, reason="Content too verbose")
            return GuardrailResult(action=GuardrailAction.PASS, content=content)

        guardrail = CustomFunctionGuardrail("complex", complex_checker)

        # Test urgent during business hours
        result = await guardrail.check("Urgent request", {"business_hours": True})
        assert result.action == GuardrailAction.PASS

        # Test urgent outside business hours
        result = await guardrail.check("Urgent request", {"business_hours": False})
        assert result.action == GuardrailAction.MODIFY
        assert "[Note: Received outside business hours]" in result.content

        # Test long content
        long_content = "x" * 51
        result = await guardrail.check(long_content, {})
        assert result.action == GuardrailAction.BLOCK
        assert "too verbose" in result.reason

        # Test normal content
        result = await guardrail.check("Normal request", {})
        assert result.action == GuardrailAction.PASS
