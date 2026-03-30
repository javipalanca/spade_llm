"""
Guardrails Example

Demonstrates input and output guardrails with two LLM agents.

Input guardrails:  keyword filter, profanity filter
Output guardrails: LLM-based safety validator

Setup:
  1. cp .env.example .env  (fill in LLM_MODEL)
  2. spade run  (in a separate terminal)
  3. python examples/guardrails.py
"""

import logging
import os
import spade

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.providers import LLMProvider
from spade_llm.guardrails import (
    KeywordGuardrail, LLMGuardrail,
    GuardrailAction, InputGuardrail, OutputGuardrail
)
from spade_llm.utils import load_env_vars
from typing import List

logging.basicConfig(level=logging.INFO)


def create_input_guardrails() -> List[InputGuardrail]:
    """Create input guardrail pipeline."""
    # 1. Keyword guardrail - blocks harmful content
    keyword_guardrail = KeywordGuardrail(
        name="harmful_content_filter",
        blocked_keywords=["hack", "exploit", "malware", "virus", "illegal", "bomb"],
        action=GuardrailAction.BLOCK,
        case_sensitive=False,
        blocked_message="I cannot help with potentially harmful activities."
    )
    
    # 2. Profanity filter - replaces inappropriate words
    profanity_guardrail = KeywordGuardrail(
        name="profanity_filter", 
        blocked_keywords=["damn", "hell", "stupid", "idiot", "crap"],
        action=GuardrailAction.MODIFY,
        replacement="[FILTERED]",
        case_sensitive=False
    )

    return [keyword_guardrail, profanity_guardrail]


def create_output_guardrails(safety_provider: LLMProvider) -> List[OutputGuardrail]:
    """Create output guardrail pipeline."""
    # LLM-based safety checker
    safety_guardrail = LLMGuardrail(
        name="llm_safety_checker",
        provider=safety_provider,
        safety_prompt="""
        Analyze this AI response for safety. Check for harmful instructions, personal attacks, 
        inappropriate content, or anything that could cause harm.
        
        Respond with JSON: {{"safe": true/false, "reason": "explanation if unsafe"}}
        
        AI Response: {content}
        """,
        blocked_message="I apologize, but I cannot provide that response due to safety concerns."
    )
    
    return [safety_guardrail]


def on_guardrail_trigger(result):
    if result.action == GuardrailAction.BLOCK:
        print(f"[GUARDRAIL BLOCKED] {result.reason}")
    elif result.action == GuardrailAction.MODIFY:
        print(f"[GUARDRAIL MODIFIED] {result.reason}")
    elif result.action == GuardrailAction.WARNING:
        print(f"[GUARDRAIL WARNING] {result.reason}")


async def main():
    load_env_vars()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")
    print("=== Guardrails Example ===")

    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    main_provider = LLMProvider(
        model=model,
        temperature=0.7,
        timeout=120.0,
    )
    safety_provider = LLMProvider(
        model=model,
        temperature=0.3,
        timeout=60.0,
    )

    input_guardrails = create_input_guardrails()
    output_guardrails = create_output_guardrails(safety_provider)

    llm_agent = LLMAgent(
        jid=f"llm_guardian@{xmpp_server}",
        password="llm_pass",
        provider=main_provider,
        system_prompt="You are a helpful AI assistant. Be concise and informative.",
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        on_guardrail_trigger=on_guardrail_trigger,
    )
    await llm_agent.start()
    print(f"LLM agent started: llm_guardian@{xmpp_server}")

    def display_response(message: str, sender: str):
        print(f"\nAssistant: {message}")

    chat = ChatAgent(
        jid=f"user@{xmpp_server}",
        password="user_pass",
        target_agent_jid=f"llm_guardian@{xmpp_server}",
        display_callback=display_response,
        verbose=False,
    )
    await chat.start()
    print(f"Chat agent started: user@{xmpp_server}")

    print("\nGuardrails active:")
    print("  Input:  keyword filter, profanity filter")
    print("  Output: LLM safety validator")
    print("Type 'exit' to quit.\n")

    await chat.run_interactive()

    await chat.stop()
    await llm_agent.stop()
    print("Agents stopped.")


if __name__ == "__main__":
    spade.run(main())