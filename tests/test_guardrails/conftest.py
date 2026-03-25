"""Fixtures for guardrails tests."""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from spade.message import Message
from spade_llm.guardrails import (
    GuardrailAction, GuardrailResult,
    KeywordGuardrail, RegexGuardrail, CustomFunctionGuardrail
)
from spade_llm.providers.base_provider import BaseLLMProvider


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for LLMGuardrail tests."""
    provider = Mock(spec=BaseLLMProvider)
    
    # Mock the get_response method to return JSON
    async def mock_get_response(context, tools=None):
        return '{"safe": true, "reason": "Content is safe"}'
    
    provider.get_response = AsyncMock(side_effect=mock_get_response)
    return provider


@pytest.fixture
def mock_unsafe_llm_provider():
    """Create a mock LLM provider that flags content as unsafe."""
    provider = Mock(spec=BaseLLMProvider)
    
    async def mock_get_response(context, tools=None):
        return '{"safe": false, "reason": "Content contains harmful material"}'
    
    provider.get_response = AsyncMock(side_effect=mock_get_response)
    return provider


@pytest.fixture
def mock_error_llm_provider():
    """Create a mock LLM provider that raises errors."""
    provider = Mock(spec=BaseLLMProvider)
    provider.get_response = AsyncMock(side_effect=Exception("LLM provider error"))
    return provider


@pytest.fixture
def basic_keyword_guardrail():
    """Create a basic keyword guardrail for testing."""
    return KeywordGuardrail(
        name="test_keyword",
        blocked_keywords=["bad", "harmful", "inappropriate"],
        action=GuardrailAction.BLOCK
    )


@pytest.fixture
def modify_keyword_guardrail():
    """Create a keyword guardrail that modifies content."""
    return KeywordGuardrail(
        name="modify_keyword",
        blocked_keywords=["replace_me", "filter_this"],
        action=GuardrailAction.MODIFY,
        replacement="[FILTERED]"
    )


@pytest.fixture
def case_sensitive_keyword_guardrail():
    """Create a case-sensitive keyword guardrail."""
    return KeywordGuardrail(
        name="case_sensitive",
        blocked_keywords=["ExactCase", "CaseSensitive"],
        action=GuardrailAction.BLOCK,
        case_sensitive=True
    )


@pytest.fixture
def email_regex_guardrail():
    """Create a regex guardrail that redacts email addresses."""
    return RegexGuardrail(
        name="email_redactor",
        patterns={
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]'
        }
    )


@pytest.fixture
def blocking_regex_guardrail():
    """Create a regex guardrail that blocks certain patterns."""
    return RegexGuardrail(
        name="pattern_blocker",
        patterns={
            r'\b\d{3}-\d{2}-\d{4}\b': GuardrailAction.BLOCK  # SSN pattern
        },
        blocked_message="Sensitive information detected"
    )


@pytest.fixture
def simple_custom_function():
    """Create a simple custom function for testing."""
    def check_length(content: str, context: Dict[str, Any]) -> GuardrailResult:
        if len(content) > 100:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Content too long"
            )
        return GuardrailResult(action=GuardrailAction.PASS, content=content)
    
    return check_length


@pytest.fixture
def modify_custom_function():
    """Create a custom function that modifies content."""
    def add_prefix(content: str, context: Dict[str, Any]) -> GuardrailResult:
        if not content.startswith("[CHECKED]"):
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                content=f"[CHECKED] {content}",
                reason="Added safety prefix"
            )
        return GuardrailResult(action=GuardrailAction.PASS, content=content)
    
    return add_prefix


@pytest.fixture
async def async_custom_function():
    """Create an async custom function for testing."""
    async def async_check(content: str, context: Dict[str, Any]) -> GuardrailResult:
        # Simulate async work
        import asyncio
        await asyncio.sleep(0.001)
        
        if "async_block" in content:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Async block triggered"
            )
        return GuardrailResult(action=GuardrailAction.PASS, content=content)
    
    return async_check


@pytest.fixture
def custom_function_guardrail(simple_custom_function):
    """Create a custom function guardrail."""
    return CustomFunctionGuardrail(
        name="length_checker",
        check_function=simple_custom_function
    )


@pytest.fixture
def test_context():
    """Create a basic test context."""
    return {
        "conversation_id": "test_conv_123",
        "sender": "user@example.com",
        "message": Mock(spec=Message)
    }


@pytest.fixture
def guardrail_result_pass():
    """Create a PASS guardrail result."""
    return GuardrailResult(
        action=GuardrailAction.PASS,
        content="safe content",
        reason="Content is safe"
    )


@pytest.fixture
def guardrail_result_block():
    """Create a BLOCK guardrail result."""
    return GuardrailResult(
        action=GuardrailAction.BLOCK,
        reason="Content blocked for safety",
        custom_message="Your content was blocked"
    )


@pytest.fixture
def guardrail_result_modify():
    """Create a MODIFY guardrail result."""
    return GuardrailResult(
        action=GuardrailAction.MODIFY,
        content="modified content",
        reason="Content was modified",
        metadata={"changes": 1}
    )


@pytest.fixture
def trigger_callback_log():
    """Create a callback function that logs trigger events."""
    log = []
    
    def callback(result: GuardrailResult):
        log.append({
            "action": result.action,
            "reason": result.reason,
            "content": result.content
        })
    
    callback.log = log
    return callback
