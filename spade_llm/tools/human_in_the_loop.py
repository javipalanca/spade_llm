"""Human-in-the-loop tool implementation for SPADE_LLM."""

import logging
from asyncio import TimeoutError
from typing import Optional

from spade.agent import Agent

from ..behaviour.human_interaction import HumanInteractionBehaviour
from .llm_tool import LLMTool

logger = logging.getLogger("spade_llm.tools.human_in_the_loop")


class HumanInTheLoopTool(LLMTool):
    """
    A tool that allows LLM agents to consult with human experts.

    This tool creates a communication channel via XMPP to ask questions
    to a designated human expert and wait for their responses.
    """

    def __init__(
        self,
        human_expert_jid: str,
        timeout: float = 300.0,
        name: str = "ask_human_expert",
        description: Optional[str] = None,
    ):
        """
        Initialize the Human-in-the-Loop tool.

        Args:
            human_expert_jid: The JID of the human expert to consult
            timeout: Maximum time to wait for human response in seconds (default: 300)
            name: Name of the tool for LLM to reference
            description: Description of when to use this tool
        """
        self.human_jid = human_expert_jid
        self.timeout = timeout
        self._agent: Optional[Agent] = None  # Will be set when added to an agent

        # Default description if not provided
        if description is None:
            description = (
                "Ask a human expert for help when you need clarification, "
                "additional information, or human judgment. Use this when "
                "you're unsure about something or need real-world context."
            )

        # Define parameters schema for the LLM
        parameters = {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to ask the human expert",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context to help the human understand the question better",
                },
            },
            "required": ["question"],
        }

        # Initialize parent with execute method
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            func=self._ask_human,  # Use internal method as the function
        )

        logger.info(f"HumanInTheLoopTool initialized with expert JID: {human_expert_jid}")

    def set_agent(self, agent: Agent):
        """
        Set the agent that owns this tool.
        This is called automatically when the tool is added to an agent.

        Args:
            agent: The SPADE agent that will use this tool
        """
        self._agent = agent
        logger.debug(f"HumanInTheLoopTool bound to agent: {agent.jid}")

    async def _ask_human(self, question: str, context: Optional[str] = None) -> str:
        """
        Execute the human consultation.

        This method is called by the LLMBehaviour when the tool is invoked.

        Args:
            question: The question to ask the human
            context: Optional additional context

        Returns:
            The human's response or an error message
        """
        if not self._agent:
            logger.error("Tool not bound to any agent")
            return "Error: This tool is not properly configured. No agent assigned."

        logger.info(f"Asking human expert: {question[:100]}...")

        interaction = HumanInteractionBehaviour(
            human_jid=self.human_jid,
            question=question,
            context=context,
            timeout=self.timeout,
        )

        self._agent.add_behaviour(interaction)

        try:
            # Wait for the behaviour to complete with timeout
            await interaction.join(timeout=self.timeout)

            # Get the response
            if interaction.response:
                logger.info("Received response from human expert")
                return interaction.response
            else:
                logger.warning("Human expert did not provide a response")
                return "The human expert did not provide a response."

        except TimeoutError:
            # The behaviour didn't complete within the timeout
            logger.warning(f"Timeout waiting for human expert response ({self.timeout}s)")

            # Clean up: remove the behaviour since it didn't complete
            self._agent.remove_behaviour(interaction)

            return (
                f"Timeout: The human expert did not respond within "
                f"{self.timeout} seconds. Please try again later or "
                f"proceed without human input."
            )
        except Exception as e:
            logger.error(f"Error during human interaction: {e}")

            # Try to clean up the behaviour
            try:
                self._agent.remove_behaviour(interaction)
            except Exception:
                pass

            return f"Error consulting human expert: {str(e)}"
