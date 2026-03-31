"""Chat Agent implementation for SPADE_LLM."""

import asyncio
import logging
import sys
from typing import Callable, Optional

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

logger = logging.getLogger("spade_llm.agent.chat")


class ChatAgent(Agent):
    """
    A SPADE agent that acts as an interface between human users and other agents.

    This agent provides a standardized way to:
    - Receive input from users and forward it to target agents
    - Receive responses from agents and display them to users
    - Handle asynchronous communication patterns with proper synchronization
    """

    class SendBehaviour(CyclicBehaviour):
        """Behaviour for sending pending messages to the target agent."""

        async def run(self):
            message_to_send = self.get("message_to_send")
            if message_to_send:
                target_jid = self.get("target_agent_jid")
                msg = Message(to=target_jid)
                msg.body = message_to_send
                msg.set_metadata("performative", "request")
                msg.set_metadata("message_type", "llm")  # Mark as LLM-targeted message

                # Log if verbose mode is enabled
                verbose = self.get("verbose")
                if verbose:
                    logger.info(f"ChatAgent sending: '{message_to_send}' to {target_jid}")

                await self.send(msg)
                self.set("message_to_send", None)

                # Call the on_message_sent callback if provided
                callback = self.get("on_message_sent")
                if callback:
                    callback(message_to_send, target_jid)

            await asyncio.sleep(0.1)

    class ReceiveBehaviour(CyclicBehaviour):
        """Behaviour for receiving and displaying messages from the target agent."""

        async def run(self):
            response = await self.receive(timeout=0.1)
            if response:
                display_callback = self.get("display_callback")
                if display_callback:
                    display_callback(response.body, str(response.sender))
                else:
                    # Default display behavior
                    print(f"\nResponse from {response.sender}: '{response.body}'")

                # Mark that response was received for synchronization
                self.set("response_received", True)

                callback = self.get("on_message_received")
                if callback:
                    callback(response.body, str(response.sender))

            await asyncio.sleep(0.1)

    def __init__(
        self,
        jid: str,
        password: str,
        target_agent_jid: str,
        display_callback: Optional[Callable[[str, str], None]] = None,
        on_message_sent: Optional[Callable[[str, str], None]] = None,
        on_message_received: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
        verify_security: bool = False,
    ):
        """
        Initialize the chat agent.

        Args:
            jid: The Jabber ID of this agent
            password: The password for this agent
            target_agent_jid: The JID of the agent to communicate with
            display_callback: Optional callback for customizing how responses are displayed
                            (receives message_body, sender_jid)
            on_message_sent: Optional callback called after sending a message
                           (receives message_body, recipient_jid)
            on_message_received: Optional callback called after receiving a message
                               (receives message_body, sender_jid)
            verbose: Whether to log detailed information
            verify_security: Whether to verify security certificates
        """
        super().__init__(jid, password, verify_security=verify_security)

        self.target_agent_jid = target_agent_jid
        self.display_callback = display_callback
        self.on_message_sent = on_message_sent
        self.on_message_received = on_message_received
        self.verbose = verbose

    async def setup(self):
        """Set up the chat agent with send and receive behaviours."""
        logger.info(f"ChatAgent {self.jid} starting, will communicate with {self.target_agent_jid}")

        # Store configuration in agent's data
        self.set("target_agent_jid", self.target_agent_jid)
        self.set("message_to_send", None)
        self.set("display_callback", self.display_callback)
        self.set("on_message_sent", self.on_message_sent)
        self.set("on_message_received", self.on_message_received)
        self.set("verbose", self.verbose)
        self.set("response_received", False)  # For synchronization

        # Add behaviours
        send_behaviour = self.SendBehaviour()
        receive_behaviour = self.ReceiveBehaviour()
        self.add_behaviour(send_behaviour)
        self.add_behaviour(receive_behaviour)

    def send_message(self, message: str):
        """
        Queue a message to be sent to the target agent.

        Args:
            message: The message to send
        """
        self.set("message_to_send", message)
        self.set("response_received", False)  # Reset for new message

    async def send_message_async(self, message: str):
        """
        Send a message to the target agent asynchronously.

        Args:
            message: The message to send
        """
        msg = Message(to=self.target_agent_jid)
        msg.body = message
        msg.set_metadata("performative", "request")
        msg.set_metadata("message_type", "llm")  # Mark as LLM-targeted message

        if self.verbose:
            logger.info(f"ChatAgent sending: '{message}' to {self.target_agent_jid}")

        await self.send(msg)

        # Call the callback if provided
        if self.on_message_sent:
            self.on_message_sent(message, self.target_agent_jid)

    async def wait_for_response(self, timeout: float = 10.0) -> bool:
        """
        Wait for a response to be received.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if response was received, False if timeout
        """
        start_time = asyncio.get_running_loop().time()

        while not self.get("response_received"):
            await asyncio.sleep(0.1)
            if asyncio.get_running_loop().time() - start_time > timeout:
                return False

        return True

    def is_waiting_response(self) -> bool:
        """
        Check if the agent is waiting for a response.

        Returns:
            True if waiting, False otherwise
        """
        return not self.get("response_received")

    async def run_interactive(
        self,
        input_prompt: str = "> ",
        exit_command: str = "exit",
        response_timeout: float = 10.0,
    ):
        """
        Run an interactive chat session with this agent.

        Args:
            input_prompt: The prompt to display for user input
            exit_command: The command to exit the chat
            response_timeout: Maximum time to wait for responses in seconds
        """
        await run_interactive_chat(
            self,
            input_prompt=input_prompt,
            exit_command=exit_command,
            response_timeout=response_timeout,
        )


# Utility function for safe async input
async def safe_input(prompt: str = "") -> str:
    """
    Execute input() in a separate thread safely.

    Args:
        prompt: The prompt to display

    Returns:
        The user's input
    """
    sys.stdout.flush()  # Clear output buffer
    return await asyncio.to_thread(input, prompt)


async def run_interactive_chat(
    chat_agent: ChatAgent,
    input_prompt: str = "> ",
    exit_command: str = "exit",
    response_timeout: float = 10.0,
):
    """
    Run an interactive chat session with the chat agent using synchronized communication.

    Args:
        chat_agent: The ChatAgent instance to use
        input_prompt: The prompt to display for user input
        exit_command: The command to exit the chat
        response_timeout: Maximum time to wait for responses in seconds
    """
    print(f"\nChat session started. Type '{exit_command}' to quit.\n")

    while True:
        try:
            # Small pause to ensure any pending output is displayed
            await asyncio.sleep(0.1)

            # Get user input safely
            user_input = await safe_input(input_prompt)

            if user_input.lower() == exit_command:
                break

            if user_input.strip():  # Only send non-empty messages
                # Send message
                chat_agent.send_message(user_input)

                # Wait for response with timeout
                response_received = await chat_agent.wait_for_response(response_timeout)

                if not response_received:
                    print("\nTimeout waiting for response")

                # Small pause before next prompt
                await asyncio.sleep(0.2)

        except KeyboardInterrupt:
            print("\nChat interrupted by user")
            break
        except Exception as e:
            print(f"Error in chat loop: {e}")
            continue

    print("\nChat session ended.")
