"""
Simple Coordinator Example

Minimal test of CoordinatorAgent: Calculator -> Reporter -> Saver.

Setup:
  1. cp .env.example .env  (fill in LLM_MODEL)
  2. spade run             (in a separate terminal)
  3. python examples/simple_coordinator.py
"""

import asyncio
import logging
import os

import spade

from spade_llm.agent import ChatAgent, LLMAgent
from spade_llm.agent.coordinator_agent import CoordinatorAgent
from spade_llm.providers import LLMProvider
from spade_llm.tools import LLMTool
from spade_llm.utils import load_env_vars

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("spade_llm").setLevel(logging.INFO)

# 1. AGENT PROMPTS
CALCULATOR_PROMPT = """You are a simple calculator agent.

When asked to calculate an expression:
1. Compute the result
2. Return ONLY the numeric answer (no explanation needed)

Examples:
- "Calculate (10 + 5) * 2" → "30"
- "What is 7 + 3?" → "10"
- "Compute 100 / 4" → "25"

Be concise and accurate.
"""

REPORTER_PROMPT = """You are a result formatting agent.

When given a calculation result to format:
1. Create a brief, clear statement with the result
2. Keep it to 1-2 sentences maximum

Examples:
- Input: "Format result: 30 for calculation (10 + 5) * 2"
  Output: "The result of (10 + 5) * 2 is 30."

- Input: "Format: 10 from 7 + 3"
  Output: "7 + 3 equals 10."

Be professional and concise.
"""


SAVER_PROMPT = """You are a storage agent that persists finalized reports.

When you receive a formatted report:
1. Call the save_report tool exactly once with the complete text to store.
2. After the tool call, acknowledge success to the coordinator (no additional formatting).

If you cannot save the file, explain why.
"""

REPORT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "coordinator_report.txt")


def _create_save_report_tool(file_path: str) -> LLMTool:
    """Create an LLM tool that saves report text to the provided path."""

    def save_report(report_text: str) -> str:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as report_file:
            report_file.write(report_text)
        return f"Report saved to {file_path}"

    return LLMTool(
        name="save_report",
        description="Persist the finalized report text to disk",
        parameters={
            "type": "object",
            "properties": {
                "report_text": {
                    "type": "string",
                    "description": "The full report text to store",
                }
            },
            "required": ["report_text"],
        },
        func=save_report,
    )


async def main():
    print("=== Simple Coordinator Example ===\n")

    load_env_vars()
    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")

    if os.path.exists(REPORT_SAVE_PATH):
        os.remove(REPORT_SAVE_PATH)

    provider = LLMProvider(
        model=model,
    )

    calculator = LLMAgent(
        jid=f"calculator@{xmpp_server}",
        password="calc_pass",
        system_prompt=CALCULATOR_PROMPT,
        provider=provider,
        verify_security=False,
    )

    reporter = LLMAgent(
        jid=f"reporter@{xmpp_server}",
        password="report_pass",
        system_prompt=REPORTER_PROMPT,
        provider=provider,
        verify_security=False,
    )

    saver = LLMAgent(
        jid=f"saver@{xmpp_server}",
        password="save_pass",
        system_prompt=SAVER_PROMPT,
        provider=provider,
        tools=[_create_save_report_tool(REPORT_SAVE_PATH)],
        verify_security=False,
    )

    coordinator = CoordinatorAgent(
        jid=f"coordinator@{xmpp_server}",
        password="coord_pass",
        subagent_ids=[
            f"calculator@{xmpp_server}",
            f"reporter@{xmpp_server}",
            f"saver@{xmpp_server}",
        ],
        coordination_session="calc_session",
        provider=provider,
        verify_security=False,
    )

    completion_detected = asyncio.Event()
    final_response = []

    def display_callback(message: str, sender: str):
        print(f"Response from {sender}:")
        print(f"   {message}")
        print()
        if "<TASK_COMPLETE>" in message or "<END>" in message or "<DONE>" in message:
            print("Task complete.")
            final_response.append(message)
            completion_detected.set()

    chat_agent = ChatAgent(
        jid=f"user@{xmpp_server}",
        password="user_pass",
        target_agent_jid=f"coordinator@{xmpp_server}",
        display_callback=display_callback,
        verify_security=False,
    )

    try:
        await calculator.start()
        await reporter.start()
        await saver.start()
        await coordinator.start()
        await chat_agent.start()

        print("All agents started.")
        await asyncio.sleep(2)

        print("\nTest: Calculate (10 + 5) * 2, format it, and save to disk.\n")

        test_request = """Please coordinate this calculation task step by step:

1. Ask the calculator agent to compute: (10 + 5) * 2
2. Ask the reporter agent to format the result nicely
3. Send the formatted output to the saver agent and have it call save_report with the exact text

Use your send_to_agent tool for each step. Work sequentially - wait for each response before proceeding.

When everything is complete, end your response with <TASK_COMPLETE>
"""

        chat_agent.send_message(test_request)
        await asyncio.sleep(1)

        print("Waiting for coordination to complete (max 60 seconds)...")

        try:
            await asyncio.wait_for(completion_detected.wait(), timeout=60)
            print("\nCoordination completed successfully.")
        except asyncio.TimeoutError:
            print("\nCoordination timed out after 60 seconds.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await chat_agent.stop()
        await coordinator.stop()
        await saver.stop()
        await reporter.stop()
        await calculator.stop()
        print("Agents stopped.")


if __name__ == "__main__":
    spade.run(main())
