"""
Document Creation Workflow

Multi-agent pipeline: Researcher -> Editor -> Reviewer -> Publisher.

Setup:
  1. cp .env.example .env  (fill in LLM_MODEL)
  2. spade run             (in a separate terminal)
  3. python examples/document_workflow.py
"""

import asyncio
import logging
import os
from datetime import datetime

import spade
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from spade_llm.agent import ChatAgent, LLMAgent
from spade_llm.providers import LLMProvider
from spade_llm.routing import RoutingResponse
from spade_llm.tools import LangChainToolAdapter
from spade_llm.utils import load_env_vars

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("spade_llm").setLevel(logging.INFO)


def review_router(msg, response, context):
    """Routes reviewer decisions to publisher, researcher, or editor."""
    response_lower = response.lower()

    domain = str(msg.sender).split("@")[1]

    if "<task_complete>" in response_lower:
        return RoutingResponse(
            recipients=f"publisher@{domain}", transform=lambda x: x.replace("<TASK_COMPLETE>", "").strip()
        )
    elif "<major_revision>" in response_lower:
        return RoutingResponse(recipients=[f"researcher@{domain}"])
    else:
        return RoutingResponse(recipients=f"editor@{domain}")


async def main():
    # Load environment
    load_env_vars()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")
    XMPP_SERVER = os.environ.get("XMPP_SERVER", "localhost")
    agents_config = {
        "researcher": (f"researcher@{XMPP_SERVER}", "Research Agent"),
        "editor": (f"editor@{XMPP_SERVER}", "Editor Agent"),
        "reviewer": (f"reviewer@{XMPP_SERVER}", "Reviewer Agent"),
        "publisher": (f"publisher@{XMPP_SERVER}", "Publisher Agent"),
        "human": (f"human@{XMPP_SERVER}", "Human Agent"),
    }

    passwords = {role: f"{role}_pass" for role in agents_config}

    # Create provider
    provider = LLMProvider(
        model=model,
    )

    tools = [
        LangChainToolAdapter(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
    ]

    # Create agents
    agents = {}

    # Research Agent
    agents["researcher"] = LLMAgent(
        jid=agents_config["researcher"][0],
        password=passwords["researcher"],
        provider=provider,
        reply_to=agents_config["editor"][0],
        system_prompt="Research topics and create initial document drafts with title, sections, and conclusion.",
        tools=tools,
    )

    # Editor Agent
    agents["editor"] = LLMAgent(
        jid=agents_config["editor"][0],
        password=passwords["editor"],
        provider=provider,
        reply_to=agents_config["reviewer"][0],
        system_prompt="Improve drafts for clarity, structure, and style. Return complete documents.",
    )

    # Reviewer Agent
    agents["reviewer"] = LLMAgent(
        jid=agents_config["reviewer"][0],
        password=passwords["reviewer"],
        provider=provider,
        routing_function=review_router,
        system_prompt="""Review documents carefully but pragmatically. Each revision represents a cost in time and resources, so find a balance between quality and efficiency.

        Choose one of these actions:
        1. If the document is ready for publication (even with acceptable minor imperfections): 
           - Include the ENTIRE DOCUMENT in your response
           - Add '<TASK_COMPLETE>' at the very end of your message
        
        2. If there are SERIOUS issues requiring additional research: 
           - Add '<MAJOR_REVISION>' and explain the problems
           - Do NOT include the full document
        
        3. If there are minor issues that don't significantly affect content quality: 
           - Briefly describe the necessary changes
           - Do NOT include the full document

        """,
        termination_markers=["<TASK_COMPLETE>"],
    )

    # Publisher Agent (simple agent that saves files)
    class PublisherAgent(spade.agent.Agent):
        async def setup(self):
            class PublishBehaviour(spade.behaviour.CyclicBehaviour):
                async def run(self):
                    msg = await self.receive(timeout=1.0)
                    if msg:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"published_document_{timestamp}.txt"
                        with open(filename, "w") as f:
                            f.write(msg.body)
                        print(f"Document saved to: {filename}")
                    await asyncio.sleep(0.1)

            self.add_behaviour(PublishBehaviour())

    agents["publisher"] = PublisherAgent(agents_config["publisher"][0], passwords["publisher"])

    # Human Agent (using ChatAgent)
    def display_callback(message, sender):
        print(f"\nDocument from {sender}:\n{message}\n")
        if "<TASK_COMPLETE>" in message:
            print("Workflow completed.\n")

    agents["human"] = ChatAgent(
        jid=agents_config["human"][0],
        password=passwords["human"],
        target_agent_jid=agents_config["researcher"][0],
        display_callback=display_callback,
    )

    # Start all agents
    for agent in agents.values():
        await agent.start()

    print("\n=== Document Creation Workflow ===")
    print("Research → Editor → Reviewer → Publisher")
    print("\nType document topics, 'exit' to quit\n")

    # Run interactive chat
    await agents["human"].run_interactive(input_prompt="Topic> ", exit_command="exit")

    # Stop all agents
    for agent in agents.values():
        await agent.stop()

    print("Workflow stopped.")


if __name__ == "__main__":
    spade.run(main())
