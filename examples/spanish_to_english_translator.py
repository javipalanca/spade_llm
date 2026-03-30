"""
Spanish to English Translator

An LLM agent that translates Spanish text to English. Terminates on non-Spanish input.

Setup:
  1. cp .env.example .env  (fill in LLM_MODEL)
  2. spade run
  3. python examples/spanish_to_english_translator.py
"""

import os

import spade

from spade_llm.agent import ChatAgent, LLMAgent
from spade_llm.providers import LLMProvider
from spade_llm.utils import load_env_vars

TRANSLATOR_PROMPT = """
You are a Spanish-to-English translator. Translate Spanish text to English.

Rules:
1. Only respond with the English translation
2. If input is NOT Spanish, respond: "This is not Spanish text. [DONE]"
3. Keep the same tone and style in translations
"""


async def main():
    load_env_vars()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")
    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    translator_jid = f"translator@{xmpp_server}"
    translator = LLMAgent(
        jid=translator_jid,
        password="translator_pass",
        provider=LLMProvider(
            model=model,
            temperature=0.3,
        ),
        system_prompt=TRANSLATOR_PROMPT,
        termination_markers=["[DONE]"],
    )

    await translator.start()
    print(f"Translator started: {translator_jid}")

    human_jid = f"human@{xmpp_server}"
    shutdown = False

    def check_response(message: str, sender: str):
        nonlocal shutdown
        print(f"\nTranslation: {message}")
        if "This is not Spanish text" in message:
            shutdown = True
            print("\nNon-Spanish detected. Shutting down...")

    chat = ChatAgent(
        jid=human_jid, password="human_pass", target_agent_jid=translator_jid, display_callback=check_response
    )

    await chat.start()
    print(f"Chat started: {human_jid}")

    print("\nType Spanish text to translate (or non-Spanish to exit)")
    print("Type 'exit' to quit\n")

    await chat.run_interactive(exit_command="exit")

    await chat.stop()
    await translator.stop()
    print("Agents stopped.")


if __name__ == "__main__":
    spade.run(main())
