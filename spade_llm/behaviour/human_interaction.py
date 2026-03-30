"""Human interaction behaviour for SPADE_LLM."""

import logging
import uuid
from typing import Optional

from spade.behaviour import OneShotBehaviour
from spade.message import Message

logger = logging.getLogger("spade_llm.behaviour.human_interaction")


class HumanInteractionBehaviour(OneShotBehaviour):
    """
    A behaviour that handles a single interaction with a human expert.

    This behaviour sends a question to a human via XMPP and waits for their response,
    using XMPP thread IDs for correlation between question and answer.
    """

    def __init__(
        self,
        human_jid: str,
        question: str,
        context: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the human interaction behaviour.

        Args:
            human_jid: The JID of the human expert to consult
            question: The question to ask the human
            context: Optional additional context to help the human understand the question
            timeout: Maximum time to wait for human response in seconds
        """
        super().__init__()
        self.human_jid = human_jid
        self.question = question
        self.context = context
        self.timeout = timeout
        self.response: Optional[str] = None
        self.query_id = str(uuid.uuid4())[:8]  # Short ID for readability

    async def run(self):
        """
        Execute the human interaction: send question and wait for response.
        """
        # Wait for agent to be fully connected to XMPP server
        # This is crucial for dynamically added behaviours
        if hasattr(self.agent, "connected_event"):
            try:
                await self.agent.connected_event.wait()
                logger.debug(f"Agent connection confirmed for query {self.query_id}")
            except Exception as e:
                logger.warning(f"Could not wait for agent connection: {e}")

        if not self.agent or not self.agent.client:
            logger.error(f"Agent XMPP client not available for query {self.query_id}")
            return

        # Send question to human
        msg = Message(to=self.human_jid)
        msg.body = self._format_question()
        msg.thread = self.query_id
        msg.set_metadata("type", "human_query")
        msg.set_metadata("query_id", self.query_id)

        logger.info(f"Sending human query {self.query_id} to {self.human_jid}")

        try:
            await self.send(msg)
            logger.info(f"Query sent successfully to {self.human_jid}")
        except Exception as e:
            logger.warning(f"Send error (message likely delivered): {e}")
            # Don't return here - continue waiting for response as message may still be delivered

        logger.debug(f"Waiting for response to query {self.query_id} (timeout: {self.timeout}s)")
        response_msg = await self.receive(timeout=self.timeout)

        if response_msg:
            self.response = response_msg.body
            logger.info(f"Received response for query {self.query_id}: {self.response[:50]}...")
        else:
            # This should not happen if join() is used with timeout
            # The timeout is handled at the join() level
            self.response = None
            logger.warning(f"No response received for query {self.query_id}")

    def _format_question(self) -> str:
        """
        Format the question for the human expert.

        Returns:
            Formatted question string
        """
        formatted = f"[Query {self.query_id}] {self.question}"

        if self.context:
            formatted += f"\n\nContext: {self.context}"

        # Add instructions for responding
        formatted += "\n\n(Please reply to this message to provide your answer)"

        return formatted
