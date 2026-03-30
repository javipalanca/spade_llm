"""Guardrail processing functions."""

import logging
from typing import Callable, List, Optional

from spade.message import Message

from .base import GuardrailAction, GuardrailResult
from .types import InputGuardrail, OutputGuardrail

logger = logging.getLogger("spade_llm.guardrails.processor")


async def apply_input_guardrails(
    content: str,
    message: Message,
    guardrails: List[InputGuardrail],
    on_trigger: Optional[Callable[[GuardrailResult], None]] = None,
    send_reply: Optional[Callable[[Message], None]] = None,
) -> Optional[str]:
    """Apply input guardrails and return processed content or None if blocked."""
    context = {
        "message": message,
        "sender": str(message.sender),
        "conversation_id": message.thread or f"{message.sender}_{message.to}",
    }

    current_content = content

    for guardrail in guardrails:
        result = await guardrail(current_content, context)

        if result.action == GuardrailAction.BLOCK:
            logger.warning(f"Input blocked by {guardrail.name}: {result.reason}")
            if on_trigger:
                on_trigger(result)

            if send_reply:
                reply = message.make_reply()
                reply.body = result.custom_message or "Your message was blocked by security filters."
                await send_reply(reply)

            return None

        elif result.action == GuardrailAction.MODIFY:
            logger.info(f"Input modified by {guardrail.name}: {result.reason}")
            current_content = result.content
            if on_trigger:
                on_trigger(result)

        elif result.action == GuardrailAction.WARNING:
            logger.warning(f"Input warning from {guardrail.name}: {result.reason}")
            if on_trigger:
                on_trigger(result)

    return current_content


async def apply_output_guardrails(
    content: str,
    original_message: Message,
    guardrails: List[OutputGuardrail],
    on_trigger: Optional[Callable[[GuardrailResult], None]] = None,
) -> str:
    """Apply output guardrails and return processed content."""
    context = {
        "original_message": original_message,
        "conversation_id": (original_message.thread or f"{original_message.sender}_{original_message.to}"),
        "llm_response": content,
    }

    current_content = content

    for guardrail in guardrails:
        result = await guardrail(current_content, context)

        if result.action == GuardrailAction.BLOCK:
            logger.warning(f"Output blocked by {guardrail.name}: {result.reason}")
            if on_trigger:
                on_trigger(result)

            return result.custom_message or "I apologize, but I cannot provide that response."

        elif result.action == GuardrailAction.MODIFY:
            logger.info(f"Output modified by {guardrail.name}: {result.reason}")
            current_content = result.content
            if on_trigger:
                on_trigger(result)

        elif result.action == GuardrailAction.WARNING:
            logger.warning(f"Output warning from {guardrail.name}: {result.reason}")
            if on_trigger:
                on_trigger(result)

    return current_content
