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

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Union

from spade.message import Message

from spade_llm.agent.llm_agent import LLMAgent
from spade_llm.context._types import _sanitize_jid_for_name
from spade_llm.context.context_manager import ContextManager
from spade_llm.routing.types import RoutingFunction, RoutingResponse
from spade_llm.tools.llm_tool import LLMTool
from spade_llm.utils import generate_conversation_id, generate_unique_id

logger = logging.getLogger("spade_llm.agent.coordinator")


class CoordinationContextManager(ContextManager):
    """
    Context manager specialized for coordination scenarios.

    Forces all subagent conversations to use the same coordination session ID,
    enabling shared context across multiple agent interactions.
    """

    def __init__(self, coordination_session: str, subagent_ids: Set[str], **kwargs):
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
        to_str = str(msg.to) if hasattr(msg, "to") else ""

        # Check if subagent
        if sender_str in self.subagent_ids or to_str in self.subagent_ids or msg.thread == self.coordination_session:
            return self.coordination_session

        # External conversation
        return generate_conversation_id(msg)

    def add_message(self, message: Message, conversation_id: Optional[str] = None) -> None:
        """Override to use coordination conversation ID logic"""
        if conversation_id is None:
            conversation_id = self._get_coordination_conversation_id(message)

        super().add_message(message, conversation_id)


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
        subagent_response_timeout: float = 120.0,
        **kwargs,
    ):
        """
        Initialize the CoordinatorAgent.

        Args:
            jid: The JID for this agent
            password: The password for this agent
            subagent_ids: List of subagent JIDs to coordinate
            coordination_session: Session ID for coordination context
            routing_function: Optional custom routing function
            subagent_response_timeout: Timeout in seconds for waiting for each subagent response
            **kwargs: Additional arguments passed to LLMAgent
        """
        # Validate inputs
        if not subagent_ids:
            raise ValueError("subagent_ids cannot be empty")

        self.subagent_ids = set(subagent_ids)
        self.coordination_session = coordination_session or generate_unique_id("coordination")
        self.subagent_response_timeout = subagent_response_timeout
        self._task_completed = False  # Flag set by complete_task tool

        if routing_function is None:
            routing_function = self._create_coordination_routing()

        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = self._default_coordination_prompt()

        coordination_context = CoordinationContextManager(
            coordination_session=self.coordination_session,
            subagent_ids=self.subagent_ids,
            system_prompt=kwargs.get("system_prompt"),
            context_management=kwargs.get("context_management", None),
        )

        kwargs["_context_override"] = coordination_context

        # Initialize tracking before calling super().__init__
        self.agent_status: Dict[str, str] = {agent_id: "idle" for agent_id in self.subagent_ids}
        self._original_requester: Optional[str] = None

        # Call parent init
        super().__init__(jid=jid, password=password, routing_function=routing_function, **kwargs)

    def _default_coordination_prompt(self) -> str:
        """Default system prompt for coordination"""
        agent_list = ", ".join(self.subagent_ids)

        return f"""You are a coordinator agent managing the following subagents: {agent_list}

COORDINATION RULES:
1. Choose the right execution strategy:
   - SEQUENTIAL: When one agent's output is needed as input for another
   - PARALLEL: When tasks are independent and can run concurrently
2. Review the full conversation context to see all agent responses
3. Only YOU can see the full coordination context - subagents only see their individual tasks
4. CRITICAL: When delegating to subagents, include ALL necessary context in your message
5. After receiving responses from all required agents, provide a final summary to the user
6. To signal completion: Call the complete_task tool

Available tools:
- send_to_agent: Delegate a task to ONE subagent and wait for response (SEQUENTIAL)
- send_to_agents_parallel: Delegate tasks to MULTIPLE subagents at once (PARALLEL)
- list_subagents: See available agents and their current status
- complete_task: Signal that coordination is complete

WORKFLOW EXAMPLES:

Sequential (dependency between agents):
1. Ask agent_A to "Calculate X"
2. Wait for result (e.g., "42")
3. Ask agent_B to "Format the number 42" (include the actual value!)

Parallel (independent tasks):
1. Use send_to_agents_parallel to delegate to multiple agents at once
2. All agents work concurrently
3. Collect and combine all responses
"""

    def _create_coordination_routing(self) -> RoutingFunction:
        """Create routing function for coordination responses"""

        def coordination_routing(msg: Message, response: str, context: Dict[str, Any]) -> Union[str, RoutingResponse]:

            sender_str = str(msg.sender)

            if sender_str not in self.subagent_ids and self._original_requester is None:
                self._original_requester = sender_str

            # Check for completion via complete_task tool
            if self._task_completed and self._original_requester is not None:
                # Store the requester to return, then reset for next coordination cycle
                original = self._original_requester
                self._task_completed = False
                self._original_requester = None
                return original

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
            self._create_send_to_agents_parallel_tool(),
            self._create_list_subagents_tool(),
            self._create_complete_task_tool(),
        ]

        for tool in coordination_tools:
            self.add_tool(tool)

    def _create_send_to_agent_tool(self) -> LLMTool:
        """Create tool for sending tasks to subagents that waits for responses"""
        agent = self

        async def send_to_agent(agent_id: str, message: str) -> str:
            """
            Send a task/message to a specific subagent and wait for response.

            This tool receives messages directly from the agent's mailbox
            to detect the response while waiting.
            """
            if agent_id not in agent.subagent_ids:
                return f"Error: {agent_id} is not a registered subagent"

            logger.info(f"Sending task to {agent_id} and waiting for response...")

            # Send message to subagent
            msg = Message(to=agent_id)
            msg.set_metadata("message_type", "llm")
            msg.set_metadata("coordination_session", agent.coordination_session)
            msg.thread = agent.coordination_session  # Force shared context
            msg.body = message

            await agent.llm_behaviour.send(msg)
            agent.agent_status[agent_id] = "working"

            # Wait for response by directly receiving from the agent's mailbox
            # This allows us to get the message before LLMBehaviour processes it
            start_time = asyncio.get_running_loop().time()

            while True:
                elapsed = asyncio.get_running_loop().time() - start_time

                if elapsed > agent.subagent_response_timeout:
                    logger.warning(
                        f"Timeout waiting for response from {agent_id} (>{agent.subagent_response_timeout}s)"
                    )
                    agent.agent_status[agent_id] = "timeout"
                    return f"Error: {agent_id} did not respond within {agent.subagent_response_timeout} seconds"

                # Try to receive a message with short timeout
                # We use the llm_behaviour's receive method to get from mailbox
                response_msg = await agent.llm_behaviour.receive(timeout=0.1)

                if response_msg:
                    sender_str = str(response_msg.sender)

                    # Check if this is from our target agent
                    if sender_str == agent_id:
                        agent.agent_status[agent_id] = "idle"
                        logger.info(f"Received response from {agent_id}: {response_msg.body[:100]}...")

                        # Add the message to context manually since we intercepted it
                        agent.context.add_message(response_msg, agent.coordination_session)

                        return f"Response from {agent_id}: {response_msg.body}"
                    else:
                        # Not from our target agent, this message needs to be processed normally
                        # We can't put it back easily, so we'll process it through the context
                        logger.debug(
                            f"Received message from {sender_str} while waiting for {agent_id}, adding to context"
                        )
                        agent.context.add_message(response_msg, response_msg.thread or agent.coordination_session)

                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.05)

        return LLMTool(
            name="send_to_agent",
            description="Delegate a task to a specific subagent and wait for their response. Use for sequential workflows where you need the result before proceeding.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "The JID of the target subagent"},
                    "message": {
                        "type": "string",
                        "description": "The task, question, or request to send to the subagent",
                    },
                },
                "required": ["agent_id", "message"],
            },
            func=send_to_agent,
        )

    def _create_send_to_agents_parallel_tool(self) -> LLMTool:
        """Create tool for sending tasks to multiple subagents in parallel"""
        agent = self

        async def send_to_agents_parallel(tasks: List[Dict[str, str]]) -> str:
            """
            Send tasks to multiple subagents in parallel and wait for all responses.

            Args:
                tasks: List of dicts with 'agent_id' and 'message' keys
            """
            # Validate all agent_ids first
            invalid_agents = [task["agent_id"] for task in tasks if task["agent_id"] not in agent.subagent_ids]
            if invalid_agents:
                return f"Error: {', '.join(invalid_agents)} are not registered subagents"

            logger.info(f"Sending parallel tasks to {len(tasks)} agents...")

            # Send all messages at once and track per-agent start times
            pending_agents: Dict[str, float] = {}  # agent_id -> start_time
            current_time = asyncio.get_running_loop().time()

            for task in tasks:
                msg = Message(to=task["agent_id"])
                msg.set_metadata("message_type", "llm")
                msg.set_metadata("coordination_session", agent.coordination_session)
                msg.thread = agent.coordination_session
                msg.body = task["message"]

                await agent.llm_behaviour.send(msg)
                agent.agent_status[task["agent_id"]] = "working"
                pending_agents[task["agent_id"]] = current_time

            # Collect all responses with per-agent timeout tracking
            responses: Dict[str, str] = {}

            while pending_agents:
                current_time = asyncio.get_running_loop().time()

                # Check for per-agent timeouts
                timed_out_agents = []
                for agent_id, start_time in pending_agents.items():
                    if current_time - start_time > agent.subagent_response_timeout:
                        timed_out_agents.append(agent_id)

                for agent_id in timed_out_agents:
                    agent.agent_status[agent_id] = "timeout"
                    responses[agent_id] = f"Error: did not respond within {agent.subagent_response_timeout} seconds"
                    del pending_agents[agent_id]
                    logger.warning(f"Timeout waiting for agent: {agent_id}")

                if not pending_agents:
                    break

                response_msg = await agent.llm_behaviour.receive(timeout=0.1)

                if response_msg:
                    sender_str = str(response_msg.sender)

                    if sender_str in pending_agents:
                        agent.agent_status[sender_str] = "idle"
                        responses[sender_str] = response_msg.body
                        del pending_agents[sender_str]
                        logger.info(f"Received parallel response from {sender_str} ({len(pending_agents)} remaining)")
                        agent.context.add_message(response_msg, agent.coordination_session)
                    else:
                        # Message from unexpected sender
                        logger.debug(f"Received message from {sender_str} while waiting for parallel responses")
                        agent.context.add_message(response_msg, response_msg.thread or agent.coordination_session)

                await asyncio.sleep(0.05)

            # Format responses in order of original tasks
            result_parts = []
            for task in tasks:
                agent_id = task["agent_id"]
                result_parts.append(f"Response from {agent_id}: {responses.get(agent_id, 'No response')}")

            return "\n\n".join(result_parts)

        return LLMTool(
            name="send_to_agents_parallel",
            description="Delegate tasks to multiple subagents in parallel and wait for all responses. Use when tasks are independent and can run concurrently for faster execution.",
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string", "description": "The JID of the target subagent"},
                                "message": {"type": "string", "description": "The task, question, or request to send"},
                            },
                            "required": ["agent_id", "message"],
                        },
                        "description": "List of tasks to delegate in parallel, each with agent_id and message",
                    }
                },
                "required": ["tasks"],
            },
            func=send_to_agents_parallel,
        )

    def _create_list_subagents_tool(self) -> LLMTool:
        """Create tool for listing subagents and their status"""
        agent = self

        def list_subagents() -> str:
            """List all subagents and their current status"""
            agent_info = []
            for agent_id in agent.subagent_ids:
                status = agent.agent_status.get(agent_id, "idle")
                agent_info.append(f"- {agent_id}: {status}")

            return f"Subagents in coordination session '{agent.coordination_session}':\n" + "\n".join(agent_info)

        return LLMTool(
            name="list_subagents",
            description="List all registered subagents and their current status (idle, working, timeout)",
            parameters={"type": "object", "properties": {}, "required": []},
            func=list_subagents,
        )

    def _create_complete_task_tool(self) -> LLMTool:
        """Create tool for signaling task completion"""
        agent = self

        def complete_task() -> str:
            """
            Signal that the coordination task is complete and provide final summary.
            This will route the response back to the original requester.
            """
            agent._task_completed = True
            logger.info("Task completed.")
            return "Task completed."

        return LLMTool(
            name="complete_task",
            description="Signal that all coordination work is finished. Call this with your final summary to send the result back to the original requester.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=complete_task,
        )
