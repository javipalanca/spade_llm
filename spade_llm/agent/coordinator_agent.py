"""
CoordinatorAgent: Specialized agent for multi-agent coordination

This module provides a CoordinatorAgent class that extends LLMAgent
to coordinate multiple subordinate agents.

Architecture:
- Coordinator uses CoordinationContextManager to see ALL coordination messages
- Subagents are regular LLMAgent instances that only see their individual tasks
- Coordinator routes work sequentially and waits for responses before proceeding
- All coordination happens in a shared session ID for the coordinator's visibility
"""

from typing import List, Optional, Dict, Set, Union, Any
import asyncio
import logging
from spade.message import Message
from spade_llm.agent.llm_agent import LLMAgent
from spade_llm.context.context_manager import ContextManager
from spade_llm.routing.types import RoutingFunction, RoutingResponse
from spade_llm.tools.llm_tool import LLMTool
from spade_llm.context._types import _sanitize_jid_for_name
from spade_llm.utils import generate_conversation_id, generate_unique_id

logger = logging.getLogger("spade_llm.agent.coordinator")


class CoordinationContextManager(ContextManager):
    """
    Context manager specialized for coordination scenarios.

    Forces all subagent conversations to use the same coordination session ID,
    enabling shared context across multiple agent interactions.
    """

    def __init__(
        self,
        coordination_session: str,
        subagent_ids: Set[str],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.coordination_session = coordination_session
        self._sanitize_jid_for_name = _sanitize_jid_for_name
        self.subagent_ids = subagent_ids

    def _get_coordination_conversation_id(self, msg: Message) -> str:
        """
        Override conversation ID logic for coordination.

        For messages involving subagents, use coordination session ID.
        For external messages, use standard logic.
        """
        sender_str = str(msg.sender)
        to_str = str(msg.to) if hasattr(msg, 'to') else ""

        # Check if subagent
        if (sender_str in self.subagent_ids or
            to_str in self.subagent_ids or
            msg.thread == self.coordination_session):
            return self.coordination_session

        # External conversation
        return generate_conversation_id(msg)

    def add_message(self, message: Message, conversation_id: Optional[str] = None) -> None:
        """Override to use coordination conversation ID logic"""
        if conversation_id is None:
            conversation_id = self._get_coordination_conversation_id(message)

        super().add_message(message, conversation_id)

    def add_coordination_command(
        self,
        target_agent: str,
        command: str,
        conversation_id: Optional[str] = None
    ) -> None:
        """Add a coordination command to the shared context"""
        if conversation_id is None:
            conversation_id = self.coordination_session

        user_msg_dict = {
            "role": "user",
            "content": f"[TO: {target_agent}] {command}",
            "name": self._sanitize_jid_for_name(target_agent)
        }

        self.add_message_dict(user_msg_dict, conversation_id)


class CoordinatorAgent(LLMAgent):
    """
    Agent specialized for coordinating multiple subordinate agents.

    Features:
    - Shared conversation context across all subagents
    - Automatic routing of subagent responses back to coordinator
    - Built-in coordination tools
    - Agent registry and status tracking
    """

    def __init__(
        self,
        jid: str,
        password: str,
        subagent_ids: List[str],
        coordination_session: Optional[str] = None,
        routing_function: Optional[RoutingFunction] = None,
        **kwargs
    ):
        # Validate inputs
        if not subagent_ids:
            raise ValueError("subagent_ids cannot be empty")

        self.subagent_ids = set(subagent_ids)
        # Generate unique session ID if not provided
        self.coordination_session = coordination_session or generate_unique_id("coordination")

        if routing_function is None:
            routing_function = self._create_coordination_routing()

        if 'system_prompt' not in kwargs:
            kwargs['system_prompt'] = self._default_coordination_prompt()

        coordination_context = CoordinationContextManager(
            coordination_session=self.coordination_session,
            subagent_ids=self.subagent_ids,
            system_prompt=kwargs.get('system_prompt'),
            context_management=kwargs.get('context_management', None)
        )

        kwargs['_context_override'] = coordination_context

        # Initialize tracking before calling super().__init__
        self.agent_status: Dict[str, str] = {
            agent_id: "unknown" for agent_id in self.subagent_ids
        }
        self._original_requester: Optional[str] = None
        self.termination_markers = ["<TASK_COMPLETE>", "<END>", "<DONE>"]
        self._response_timeout = 30.0  # Timeout for waiting for subagent responses

        # Call parent init
        super().__init__(
            jid=jid,
            password=password,
            routing_function=routing_function,
            **kwargs
        )

    def _default_coordination_prompt(self) -> str:
        """Default system prompt for coordination"""
        agent_list = ", ".join(self.subagent_ids)
        return f"""You are a coordinator agent managing the following subagents: {agent_list}

COORDINATION RULES:
1. Work SEQUENTIALLY - wait for each agent to respond before sending the next command
2. Review the full conversation context to see all agent responses
3. Only YOU can see the full coordination context - subagents only see their individual tasks
4. CRITICAL: When sending commands to subagents, include ALL necessary context and information in the command itself. Subagents cannot see previous messages or responses from other agents.
5. After receiving responses from all required agents, provide a final summary to the user
6. Use termination markers (<TASK_COMPLETE>, <END>, or <DONE>) ONLY when all work is finished

Available tools:
- send_to_agent: Send a command to a specific subagent
- list_subagents: See available agents and their current status

WORKFLOW:
1. Send command to first agent (include all necessary context)
2. WAIT for response (it will appear in your context)
3. Review response, then send to next agent (include results from previous steps)
4. Repeat until all tasks complete
5. Provide final summary with termination marker

Example: If agent A returns "42", and you need agent B to format it, send:
"Format the result: 42" (not just "Format the result")

Coordination session: {self.coordination_session}
"""

    def _create_coordination_routing(self) -> RoutingFunction:
        """Create routing function for coordination responses"""
        def coordination_routing(
            msg: Message,
            response: str,
            context: Dict[str, Any]
        ) -> Union[str, RoutingResponse]:

            sender_str = str(msg.sender)

            if sender_str not in self.subagent_ids and self._original_requester is None:
                self._original_requester = sender_str

            is_completion = any(marker in response for marker in self.termination_markers)

            if is_completion and self._original_requester is not None:
                return self._original_requester

            if sender_str in self.subagent_ids:
                return str(self.jid)

            # External messages: route back to sender
            return str(msg.sender)

        return coordination_routing

    async def setup(self):
        """Override setup to add coordination tools"""
        await super().setup()

        coordination_tools = [
            self._create_send_to_agent_tool(),
            self._create_list_subagents_tool()
        ]

        for tool in coordination_tools:
            self.add_tool(tool)

    def _create_send_to_agent_tool(self) -> LLMTool:
        """Create tool for sending commands to subagents that waits for responses"""
        agent = self

        async def send_to_agent(agent_id: str, command: str) -> str:
            """
            Send a command to a specific subagent and wait for response.

            This tool receives messages directly from the agent's mailbox
            to detect the response while waiting.
            """
            if agent_id not in agent.subagent_ids:
                return f"Error: {agent_id} is not a registered subagent"

            logger.info(f"Sending command to {agent_id} and waiting for response...")

            # Send message to subagent
            msg = Message(to=agent_id)
            msg.set_metadata("message_type", "llm")
            msg.set_metadata("coordination_session", agent.coordination_session)
            msg.thread = agent.coordination_session  # Force shared context
            msg.body = command

            await agent.llm_behaviour.send(msg)
            agent.agent_status[agent_id] = "sent_command"

            # Wait for response by directly receiving from the agent's mailbox
            # This allows us to get the message before LLMBehaviour processes it
            start_time = asyncio.get_event_loop().time()

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                if elapsed > agent._response_timeout:
                    logger.warning(f"Timeout waiting for response from {agent_id} (>{agent._response_timeout}s)")
                    agent.agent_status[agent_id] = "timeout"
                    return f"Error: {agent_id} did not respond within {agent._response_timeout} seconds"

                # Try to receive a message with short timeout
                # We use the llm_behaviour's receive method to get from mailbox
                response_msg = await agent.llm_behaviour.receive(timeout=0.1)

                if response_msg:
                    sender_str = str(response_msg.sender)

                    # Check if this is from our target agent
                    if sender_str == agent_id:
                        agent.agent_status[agent_id] = "responded"
                        logger.info(f"Received response from {agent_id}: {response_msg.body[:100]}...")

                        # Add the message to context manually since we intercepted it
                        agent.context.add_message(response_msg, agent.coordination_session)

                        return f"Response from {agent_id}: {response_msg.body}"
                    else:
                        # Not from our target agent, this message needs to be processed normally
                        # We can't put it back easily, so we'll process it through the context
                        logger.debug(f"Received message from {sender_str} while waiting for {agent_id}, adding to context")
                        agent.context.add_message(response_msg, response_msg.thread or agent.coordination_session)

                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.05)

        return LLMTool(
            name="send_to_agent",
            description="Send a command to a specific subagent and wait for their response (waits by polling context)",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The JID of the target subagent"
                    },
                    "command": {
                        "type": "string",
                        "description": "The command to send to the subagent"
                    }
                },
                "required": ["agent_id", "command"]
            },
            func=send_to_agent
        )

    def _create_list_subagents_tool(self) -> LLMTool:
        """Create tool for listing subagents and their status"""
        agent = self

        def list_subagents() -> str:
            """List all subagents and their current status"""
            agent_info = []
            for agent_id in agent.subagent_ids:
                status = agent.agent_status.get(agent_id, "unknown")
                agent_info.append(f"- {agent_id}: {status}")

            return f"Subagents in coordination session '{agent.coordination_session}':\n" + "\n".join(agent_info)

        return LLMTool(
            name="list_subagents",
            description="List all registered subagents and their current status",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=list_subagents
        )

