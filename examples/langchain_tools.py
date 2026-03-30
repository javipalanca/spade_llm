"""
LangChain Tools Example

Demonstrates using LangChain tools (DuckDuckGo search, Wikipedia) with SPADE agents.

Setup:
  1. cp .env.example .env  (fill in LLM_MODEL)
  2. spade run             (in a separate terminal)
  3. python examples/langchain_tools.py
"""

import os

import spade
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from spade_llm.agent import ChatAgent, LLMAgent
from spade_llm.providers import LLMProvider
from spade_llm.tools import LangChainToolAdapter
from spade_llm.utils import load_env_vars


async def main():
    load_env_vars()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")
    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    tools = [
        LangChainToolAdapter(DuckDuckGoSearchRun()),
        LangChainToolAdapter(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
    ]

    # Option 1: use model from .env (LiteLLM reads OPENAI_API_KEY and OPENAI_API_BASE)
    provider = LLMProvider(model=model)

    # Option 2: explicit provider (overrides .env)
    # provider = LLMProvider(model="openai/gpt-4o-mini")
    # provider = LLMProvider(model="anthropic/claude-3-5-haiku-20241022")

    smart_agent = LLMAgent(
        jid=f"smart@{xmpp_server}",
        password="smart_pass",
        provider=provider,
        system_prompt="You are a helpful assistant with web search and Wikipedia access. Use them for up-to-date information.",
        tools=tools,
    )

    await smart_agent.start()
    print(f"Smart agent started: smart@{xmpp_server}")

    chat = ChatAgent(
        jid=f"human@{xmpp_server}",
        password="human_pass",
        target_agent_jid=f"smart@{xmpp_server}",
    )

    await chat.start()
    print(f"Chat agent started: human@{xmpp_server}")
    print("Available tools: DuckDuckGo Search, Wikipedia")
    print("Type 'exit' to quit.\n")

    await chat.run_interactive()

    await chat.stop()
    await smart_agent.stop()
    print("Agents stopped.")


if __name__ == "__main__":
    spade.run(main())
