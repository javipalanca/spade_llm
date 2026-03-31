"""Specific guardrail types for input and output processing."""

from typing import Any, Dict, List, Optional

from .base import Guardrail, GuardrailAction, GuardrailResult


class InputGuardrail(Guardrail):
    """Guardrail specifically for processing inputs before sending to LLM."""

    def __init__(self, name: str, enabled: bool = True, blocked_message: Optional[str] = None):
        """
        Initialize an input guardrail.

        Args:
            name: Name of the guardrail
            enabled: Whether the guardrail is active
            blocked_message: Custom message to return when input is blocked
        """
        super().__init__(
            name,
            enabled,
            blocked_message or "Your message was blocked by security filters.",
        )


class OutputGuardrail(Guardrail):
    """Guardrail specifically for processing outputs from LLM."""

    def __init__(self, name: str, enabled: bool = True, blocked_message: Optional[str] = None):
        """
        Initialize an output guardrail.

        Args:
            name: Name of the guardrail
            enabled: Whether the guardrail is active
            blocked_message: Custom message to return when output is blocked
        """
        super().__init__(
            name,
            enabled,
            blocked_message or "I apologize, but I cannot provide that response.",
        )


class CompositeGuardrail(Guardrail):
    """Allows combining multiple guardrails in sequence."""

    def __init__(
        self,
        name: str,
        guardrails: List[Guardrail],
        stop_on_block: bool = True,
        enabled: bool = True,
        blocked_message: Optional[str] = None,
    ):
        """
        Initialize a composite guardrail.

        Args:
            name: Name of the composite guardrail
            guardrails: List of guardrails to apply in sequence
            stop_on_block: Whether to stop processing on first block
            enabled: Whether the guardrail is active
            blocked_message: Custom message to return when blocked
        """
        super().__init__(name, enabled, blocked_message)
        self.guardrails = guardrails
        self.stop_on_block = stop_on_block

    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """
        Apply all guardrails in sequence.

        Args:
            content: The content to check
            context: Additional context information

        Returns:
            GuardrailResult from the combined guardrail checks
        """
        current_content = content
        accumulated_reasons = []

        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            result = await guardrail(current_content, context)

            if result.action == GuardrailAction.BLOCK:
                if self.stop_on_block:
                    # Use the composite's blocked message if set, otherwise use the individual guardrail's
                    result.custom_message = self.blocked_message or result.custom_message
                    return result
                else:
                    accumulated_reasons.append(f"{guardrail.name}: {result.reason}")

            elif result.action == GuardrailAction.MODIFY:
                current_content = result.content
                accumulated_reasons.append(f"{guardrail.name}: {result.reason}")

            elif result.action == GuardrailAction.WARNING:
                accumulated_reasons.append(f"{guardrail.name}: {result.reason}")

        # If content was modified, return MODIFY action
        if current_content != content:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                content=current_content,
                reason="; ".join(accumulated_reasons) if accumulated_reasons else None,
            )

        # Otherwise, PASS
        return GuardrailResult(action=GuardrailAction.PASS, content=current_content)
