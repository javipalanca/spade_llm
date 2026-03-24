"""
Multi-Provider Chat Example

Shows how to configure different LLM providers with SPADE agents.
Uncomment the provider block you want, or use LLM_MODEL from .env.

Setup:
  1. cp examples/.env.example .env  (configure provider)
  2. spade run             (in a separate terminal)
  3. python examples/multi_provider_chat_example.py
"""

import logging
import os
import spade

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.providers import LLMProvider
from spade_llm.utils import load_env_vars

logging.basicConfig(level=logging.WARNING)


async def main():
    load_env_vars()

    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy examples/.env.example to .env and configure it.")

    # Uses LLM_MODEL from .env; LiteLLM reads OPENAI_API_KEY and OPENAI_API_BASE automatically.
    provider = LLMProvider(model=model, temperature=0.7)

    # Explicit provider examples (override LLM_MODEL):
    # provider = LLMProvider(model="openai/gpt-4o-mini")
    # provider = LLMProvider(model="anthropic/claude-3-5-haiku-20241022")
    # provider = LLMProvider(model="openai/your-local-model", base_url="http://localhost:1234/v1")

    system_prompt = "You are a helpful AI assistant. Be concise but informative."

    smart_agent = LLMAgent(
        jid=f"smart@{xmpp_server}",
        password="smart_pass",
        provider=provider,
        system_prompt=system_prompt,
        max_interactions_per_conversation=10,
    )
    await smart_agent.start()
    print(f"Smart agent started: smart@{xmpp_server}")

    def display_response(message: str, sender: str):
        print(f"\nAssistant: {message}")

    def on_send(message: str, recipient: str):
        print(f"You: {message}")

    chat_agent = ChatAgent(
        jid=f"human@{xmpp_server}",
        password="human_pass",
        target_agent_jid=f"smart@{xmpp_server}",
        display_callback=display_response,
        on_message_sent=on_send,
        verbose=False,
    )
    await chat_agent.start()
    print(f"Chat agent started: human@{xmpp_server}")

    print("\nType 'exit' to quit.\n")

    await chat_agent.run_interactive()

    await chat_agent.stop()
    await smart_agent.stop()
    print("Agents stopped.")


if __name__ == "__main__":
    spade.run(main())