"""
Tool Calling Example

Demonstrates tool calling with SPADE agents.

Setup:
  1. cp examples/.env.example .env  (fill in LLM_MODEL)
  2. spade run  (in a separate terminal)
  3. python examples/ollama_with_tools.py
"""

import os
from datetime import datetime
import spade

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.providers import LLMProvider
from spade_llm.tools import LLMTool
from spade_llm.utils import load_env_vars


# Simple tool functions
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str) -> str:
    """Get simulated weather for a city."""
    weather_data = {
        "madrid": "22C, sunny",
        "london": "15C, cloudy",
        "new york": "18C, rainy",
        "tokyo": "25C, clear",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


async def main():
    load_env_vars()
    print("=== Tool Calling Example ===\n")

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy examples/.env.example to .env and configure it.")
    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    tools = [
        LLMTool(
            name="get_current_time",
            description="Get current date and time",
            parameters={"type": "object", "properties": {}, "required": []},
            func=get_current_time,
        ),
        LLMTool(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            func=get_weather,
        ),
    ]

    llm_agent = LLMAgent(
        jid=f"llm_agent@{xmpp_server}",
        password="llm_pass",
        provider=LLMProvider(model=model),
        system_prompt="You are a helpful assistant. Use tools to answer questions about time and weather.",
        tools=tools,
    )

    await llm_agent.start()
    print(f"LLM agent started: llm_agent@{xmpp_server}")

    chat = ChatAgent(
        jid=f"user@{xmpp_server}",
        password="user_pass",
        target_agent_jid=f"llm_agent@{xmpp_server}",
    )

    await chat.start()
    print(f"Chat agent started: user@{xmpp_server}")

    print("\nTry: 'What time is it?' or 'Weather in Madrid'")
    print("Type 'exit' to quit.\n")

    await chat.run_interactive()

    await chat.stop()
    await llm_agent.stop()
    print("Agents stopped.")


if __name__ == "__main__":
    spade.run(main())