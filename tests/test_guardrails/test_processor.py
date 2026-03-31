"""Tests for guardrail processor functions."""

from unittest.mock import AsyncMock, Mock

import pytest
from spade.message import Message

from spade_llm.guardrails.base import GuardrailAction, GuardrailResult
from spade_llm.guardrails.processor import apply_input_guardrails, apply_output_guardrails
from spade_llm.guardrails.types import InputGuardrail, OutputGuardrail


class MockInputGuardrail(InputGuardrail):
    """Mock input guardrail for testing."""

    def __init__(self, name: str, result: GuardrailResult, **kwargs):
        super().__init__(name, **kwargs)
        self.result = result
        self.call_log = []

    async def check(self, content: str, context: dict) -> GuardrailResult:
        self.call_log.append({"content": content, "context": context})
        return self.result


class MockOutputGuardrail(OutputGuardrail):
    """Mock output guardrail for testing."""

    def __init__(self, name: str, result: GuardrailResult, **kwargs):
        super().__init__(name, **kwargs)
        self.result = result
        self.call_log = []

    async def check(self, content: str, context: dict) -> GuardrailResult:
        self.call_log.append({"content": content, "context": context})
        return self.result


class TestApplyInputGuardrails:
    """Test apply_input_guardrails function."""

    @pytest.mark.asyncio
    async def test_no_guardrails(self, mock_message):
        """Test with empty guardrails list."""
        result = await apply_input_guardrails(content="Test content", message=mock_message, guardrails=[])

        assert result == "Test content"

    @pytest.mark.asyncio
    async def test_all_pass(self, mock_message):
        """Test when all guardrails pass."""
        guardrail1 = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.PASS, content="content"))
        guardrail2 = MockInputGuardrail("g2", GuardrailResult(action=GuardrailAction.PASS, content="content"))

        result = await apply_input_guardrails(
            content="Test content", message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result == "Test content"
        assert len(guardrail1.call_log) == 1
        assert len(guardrail2.call_log) == 1

    @pytest.mark.asyncio
    async def test_first_blocks(self, mock_message):
        """Test when first guardrail blocks."""
        guardrail1 = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.BLOCK, reason="blocked"))
        guardrail2 = MockInputGuardrail("g2", GuardrailResult(action=GuardrailAction.PASS))

        result = await apply_input_guardrails(
            content="Test content", message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result is None  # Blocked
        assert len(guardrail1.call_log) == 1
        assert len(guardrail2.call_log) == 0  # Should not be called

    @pytest.mark.asyncio
    async def test_modify_chain(self, mock_message):
        """Test chain of modifications."""
        guardrail1 = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.MODIFY, content="step1"))
        guardrail2 = MockInputGuardrail("g2", GuardrailResult(action=GuardrailAction.MODIFY, content="step2"))
        guardrail3 = MockInputGuardrail("g3", GuardrailResult(action=GuardrailAction.PASS, content="step2"))

        result = await apply_input_guardrails(
            content="original", message=mock_message, guardrails=[guardrail1, guardrail2, guardrail3]
        )

        assert result == "step2"

        # Verify content flows through correctly
        assert guardrail1.call_log[0]["content"] == "original"
        assert guardrail2.call_log[0]["content"] == "step1"
        assert guardrail3.call_log[0]["content"] == "step2"

    @pytest.mark.asyncio
    async def test_warning_continues_processing(self, mock_message):
        """Test that warnings don't stop processing."""
        guardrail1 = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.WARNING, reason="warning"))
        guardrail2 = MockInputGuardrail("g2", GuardrailResult(action=GuardrailAction.PASS, content="final"))

        result = await apply_input_guardrails(
            content="Test content", message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result == "Test content"  # Should continue
        assert len(guardrail1.call_log) == 1
        assert len(guardrail2.call_log) == 1

    @pytest.mark.asyncio
    async def test_context_creation(self, mock_message):
        """Test that context is properly created."""
        guardrail = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.PASS))

        await apply_input_guardrails(content="Test content", message=mock_message, guardrails=[guardrail])

        context = guardrail.call_log[0]["context"]
        assert context["message"] == mock_message
        assert context["sender"] == str(mock_message.sender)
        assert "conversation_id" in context

    @pytest.mark.asyncio
    async def test_conversation_id_from_thread(self):
        """Test conversation ID creation from message thread."""
        msg = Mock(spec=Message)
        msg.thread = "custom_thread_123"
        msg.sender = "user@example.com"
        msg.to = "bot@example.com"

        guardrail = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.PASS))

        await apply_input_guardrails(content="Test", message=msg, guardrails=[guardrail])

        context = guardrail.call_log[0]["context"]
        assert context["conversation_id"] == "custom_thread_123"

    @pytest.mark.asyncio
    async def test_conversation_id_fallback(self):
        """Test conversation ID fallback when no thread."""
        msg = Mock(spec=Message)
        msg.thread = None
        msg.sender = "user@example.com"
        msg.to = "bot@example.com"

        guardrail = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.PASS))

        await apply_input_guardrails(content="Test", message=msg, guardrails=[guardrail])

        context = guardrail.call_log[0]["context"]
        assert context["conversation_id"] == "user@example.com_bot@example.com"

    @pytest.mark.asyncio
    async def test_trigger_callback_called(self, mock_message, trigger_callback_log):
        """Test that trigger callback is called for all actions."""
        guardrail1 = MockInputGuardrail(
            "g1", GuardrailResult(action=GuardrailAction.MODIFY, content="mod", reason="modified")
        )
        guardrail2 = MockInputGuardrail("g2", GuardrailResult(action=GuardrailAction.WARNING, reason="warning"))

        await apply_input_guardrails(
            content="Test", message=mock_message, guardrails=[guardrail1, guardrail2], on_trigger=trigger_callback_log
        )

        assert len(trigger_callback_log.log) == 2
        assert trigger_callback_log.log[0]["action"] == GuardrailAction.MODIFY
        assert trigger_callback_log.log[1]["action"] == GuardrailAction.WARNING

    @pytest.mark.asyncio
    async def test_send_reply_default_message(self, mock_message):
        """Test default block message when no custom message."""
        guardrail = MockInputGuardrail("g1", GuardrailResult(action=GuardrailAction.BLOCK))

        send_reply_mock = AsyncMock()

        await apply_input_guardrails(
            content="Test", message=mock_message, guardrails=[guardrail], send_reply=send_reply_mock
        )

        call_args = send_reply_mock.call_args[0][0]
        assert "blocked by security filters" in call_args.body.lower()


class TestApplyOutputGuardrails:
    """Test apply_output_guardrails function."""

    @pytest.mark.asyncio
    async def test_no_guardrails(self, mock_message):
        """Test with empty guardrails list."""
        result = await apply_output_guardrails(content="LLM response", original_message=mock_message, guardrails=[])

        assert result == "LLM response"

    @pytest.mark.asyncio
    async def test_all_pass(self, mock_message):
        """Test when all guardrails pass."""
        guardrail1 = MockOutputGuardrail("g1", GuardrailResult(action=GuardrailAction.PASS))
        guardrail2 = MockOutputGuardrail("g2", GuardrailResult(action=GuardrailAction.PASS))

        result = await apply_output_guardrails(
            content="LLM response", original_message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result == "LLM response"
        assert len(guardrail1.call_log) == 1
        assert len(guardrail2.call_log) == 1

    @pytest.mark.asyncio
    async def test_first_blocks(self, mock_message):
        """Test when first guardrail blocks."""
        guardrail1 = MockOutputGuardrail(
            "g1", GuardrailResult(action=GuardrailAction.BLOCK, custom_message="Response blocked")
        )
        guardrail2 = MockOutputGuardrail("g2", GuardrailResult(action=GuardrailAction.PASS))

        result = await apply_output_guardrails(
            content="LLM response", original_message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result == "I apologize, but I cannot provide that response."
        assert len(guardrail1.call_log) == 1
        assert len(guardrail2.call_log) == 0  # Should not be called after block

    @pytest.mark.asyncio
    async def test_block_default_message(self, mock_message):
        """Test default message when blocking without custom message."""
        guardrail = MockOutputGuardrail("g1", GuardrailResult(action=GuardrailAction.BLOCK))

        result = await apply_output_guardrails(
            content="LLM response", original_message=mock_message, guardrails=[guardrail]
        )

        assert "cannot provide that response" in result.lower()

    @pytest.mark.asyncio
    async def test_modify_chain(self, mock_message):
        """Test chain of modifications."""
        guardrail1 = MockOutputGuardrail("g1", GuardrailResult(action=GuardrailAction.MODIFY, content="modified1"))
        guardrail2 = MockOutputGuardrail("g2", GuardrailResult(action=GuardrailAction.MODIFY, content="modified2"))

        result = await apply_output_guardrails(
            content="original response", original_message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result == "modified2"

        # Verify content flows through
        assert guardrail1.call_log[0]["content"] == "original response"
        assert guardrail2.call_log[0]["content"] == "modified1"

    @pytest.mark.asyncio
    async def test_warning_continues(self, mock_message):
        """Test that warnings don't stop processing."""
        guardrail1 = MockOutputGuardrail("g1", GuardrailResult(action=GuardrailAction.WARNING))
        guardrail2 = MockOutputGuardrail("g2", GuardrailResult(action=GuardrailAction.PASS))

        result = await apply_output_guardrails(
            content="response", original_message=mock_message, guardrails=[guardrail1, guardrail2]
        )

        assert result == "response"
        assert len(guardrail1.call_log) == 1
        assert len(guardrail2.call_log) == 1

    @pytest.mark.asyncio
    async def test_context_creation(self, mock_message):
        """Test that output context is properly created."""
        guardrail = MockOutputGuardrail("g1", GuardrailResult(action=GuardrailAction.PASS))

        await apply_output_guardrails(content="LLM response", original_message=mock_message, guardrails=[guardrail])

        context = guardrail.call_log[0]["context"]
        assert context["original_message"] == mock_message
        assert context["llm_response"] == "LLM response"
        assert "conversation_id" in context

    @pytest.mark.asyncio
    async def test_trigger_callback_called(self, mock_message, trigger_callback_log):
        """Test that trigger callback is called."""
        guardrail1 = MockOutputGuardrail(
            "g1", GuardrailResult(action=GuardrailAction.MODIFY, content="mod", reason="modified")
        )
        guardrail2 = MockOutputGuardrail("g2", GuardrailResult(action=GuardrailAction.WARNING, reason="warning"))

        await apply_output_guardrails(
            content="response",
            original_message=mock_message,
            guardrails=[guardrail1, guardrail2],
            on_trigger=trigger_callback_log,
        )

        assert len(trigger_callback_log.log) == 2
        assert trigger_callback_log.log[0]["action"] == GuardrailAction.MODIFY
        assert trigger_callback_log.log[1]["action"] == GuardrailAction.WARNING

    @pytest.mark.asyncio
    async def test_trigger_callback_on_block(self, mock_message, trigger_callback_log):
        """Test trigger callback is called on block."""
        guardrail = MockOutputGuardrail("g1", GuardrailResult(action=GuardrailAction.BLOCK, reason="unsafe"))

        await apply_output_guardrails(
            content="response", original_message=mock_message, guardrails=[guardrail], on_trigger=trigger_callback_log
        )

        assert len(trigger_callback_log.log) == 1
        assert trigger_callback_log.log[0]["action"] == GuardrailAction.BLOCK
        assert trigger_callback_log.log[0]["reason"] == "unsafe"


class TestProcessorIntegration:
    """Integration tests for processor functions."""

    @pytest.mark.asyncio
    async def test_input_output_pipeline(self, mock_message):
        """Test complete input -> LLM -> output pipeline simulation."""
        # Input guardrails
        input_guardrail = MockInputGuardrail(
            "input", GuardrailResult(action=GuardrailAction.MODIFY, content="[SAFE] original input")
        )

        # Output guardrails
        output_guardrail = MockOutputGuardrail(
            "output", GuardrailResult(action=GuardrailAction.MODIFY, content="Safe LLM response [VERIFIED]")
        )

        # Simulate input processing
        processed_input = await apply_input_guardrails(
            content="original input", message=mock_message, guardrails=[input_guardrail]
        )

        assert processed_input == "[SAFE] original input"

        # Simulate LLM processing (would happen in between)
        llm_response = "LLM response based on: " + processed_input

        # Simulate output processing
        final_output = await apply_output_guardrails(
            content=llm_response, original_message=mock_message, guardrails=[output_guardrail]
        )

        assert final_output == "Safe LLM response [VERIFIED]"

    @pytest.mark.asyncio
    async def test_input_blocked_stops_pipeline(self, mock_message):
        """Test that input blocking stops the entire pipeline."""
        input_guardrail = MockInputGuardrail("input", GuardrailResult(action=GuardrailAction.BLOCK))

        processed_input = await apply_input_guardrails(
            content="malicious input", message=mock_message, guardrails=[input_guardrail]
        )

        assert processed_input is None  # Pipeline should stop here

    @pytest.mark.asyncio
    async def test_complex_multi_guardrail_scenario(self, mock_message, trigger_callback_log):
        """Test complex scenario with multiple guardrails and actions."""
        # Multiple input guardrails with different behaviors
        input_guardrails = [
            MockInputGuardrail(
                "sanitizer",
                GuardrailResult(
                    action=GuardrailAction.MODIFY, content="sanitized input", reason="Removed harmful content"
                ),
            ),
            MockInputGuardrail(
                "warner", GuardrailResult(action=GuardrailAction.WARNING, reason="Potentially suspicious content")
            ),
            MockInputGuardrail("validator", GuardrailResult(action=GuardrailAction.PASS)),
        ]

        # Multiple output guardrails
        output_guardrails = [
            MockOutputGuardrail(
                "content_filter",
                GuardrailResult(
                    action=GuardrailAction.MODIFY, content="filtered response", reason="Filtered sensitive information"
                ),
            ),
            MockOutputGuardrail("quality_check", GuardrailResult(action=GuardrailAction.PASS)),
        ]

        # Process input
        processed_input = await apply_input_guardrails(
            content="original input", message=mock_message, guardrails=input_guardrails, on_trigger=trigger_callback_log
        )

        assert processed_input == "sanitized input"

        # Process output
        final_output = await apply_output_guardrails(
            content="LLM response",
            original_message=mock_message,
            guardrails=output_guardrails,
            on_trigger=trigger_callback_log,
        )

        assert final_output == "filtered response"

        # Verify all triggers were called
        assert len(trigger_callback_log.log) == 3  # modify, warning, modify
        assert trigger_callback_log.log[0]["action"] == GuardrailAction.MODIFY
        assert trigger_callback_log.log[1]["action"] == GuardrailAction.WARNING
        assert trigger_callback_log.log[2]["action"] == GuardrailAction.MODIFY
