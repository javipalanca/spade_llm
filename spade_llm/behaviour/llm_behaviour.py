"""LLM Behaviour implementation for SPADE agents."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from ..context import (
    ContextManager,
    create_assistant_tool_call_message,
)
from ..utils import generate_conversation_id
from ..guardrails import (
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    apply_input_guardrails,
    apply_output_guardrails,
)
from ..providers.base_provider import BaseLLMProvider
from ..routing import RoutingFunction, RoutingResponse
from ..tools import LLMTool

logger = logging.getLogger("spade_llm.behaviour")


class ConversationState:
    """Represents the state of a conversation or task."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    MAX_INTERACTIONS_REACHED = "max_interactions_reached"


class LLMBehaviour(CyclicBehaviour):
    """
    A specialized behaviour for SPADE agents that integrates Large Language Models.

    This behaviour extends the CyclicBehaviour to:
    - Receive XMPP messages
    - Process and update the conversation context
    - Send prompts to LLM providers
    - Handle tool invocation
    - Send responses back with optional conditional routing
    - Apply input and output guardrails for content filtering

    The behaviour remains active throughout the agent's lifecycle,
    but individual conversations can be terminated based on conditions.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        reply_to: Optional[str] = None,
        routing_function: Optional[RoutingFunction] = None,
        context_manager: Optional[ContextManager] = None,
        termination_markers: Optional[List[str]] = None,
        max_interactions_per_conversation: Optional[int] = None,
        on_conversation_end: Optional[Callable[[str, str], None]] = None,
        tools: Optional[List[LLMTool]] = None,
        input_guardrails: Optional[List[InputGuardrail]] = None,
        output_guardrails: Optional[List[OutputGuardrail]] = None,
        on_guardrail_trigger: Optional[Callable[[GuardrailResult], None]] = None,
        interaction_memory=None,
    ):
        """
        Initialize the LLM behaviour.

        Args:
            llm_provider: Provider for LLM integration (OpenAI, Gemini, etc.)
            reply_to: JID to send responses to. If None the reply is to the original sender
            routing_function: Optional function for conditional routing based on response content
            context_manager: Manager for conversation context (optional, will create new if None)
            termination_markers: List of strings that indicate conversation completion when present in LLM responses
            max_interactions_per_conversation: Maximum number of back-and-forth exchanges in a conversation
            on_conversation_end: Callback function when a conversation ends (receives conversation_id and reason)
            tools: Optional list of tools that can be used by the LLM
            input_guardrails: List of guardrails to apply to incoming messages
            output_guardrails: List of guardrails to apply to LLM responses
            on_guardrail_trigger: Callback when a guardrail blocks/modifies content
            interaction_memory: Optional AgentInteractionMemory instance for agent-to-agent memory
        """
        super().__init__()
        self.provider = llm_provider
        self.context = context_manager or ContextManager()
        self.reply_to = reply_to
        self.routing_function = routing_function

        if self.routing_function and self.reply_to:
            logger.info(
                "Both routing_function and reply_to provided. routing_function will take precedence."
            )

        # Conversation lifecycle management
        self.termination_markers = termination_markers or [
            "<TASK_COMPLETE>",
            "<END>",
            "<DONE>",
        ]
        self.max_interactions_per_conversation = max_interactions_per_conversation
        self.on_conversation_end = on_conversation_end

        # Store tools at the behaviour level
        self.tools: List[LLMTool] = tools or []

        # Guardrails
        self.input_guardrails: List[InputGuardrail] = input_guardrails or []
        self.output_guardrails: List[OutputGuardrail] = output_guardrails or []
        self.on_guardrail_trigger = on_guardrail_trigger

        # Interaction memory
        self.interaction_memory = interaction_memory

        # Track active conversations
        self._active_conversations: Dict[str, Dict[str, Any]] = {}
        self._processed_messages: Set[str] = set()

    async def run(self):
        """
        Main execution loop for the behaviour.
        Waits for messages, processes them with the LLM, and sends responses.
        The behaviour itself continues indefinitely, but individual conversations
        can be terminated based on configured conditions.
        """
        # Wait for incoming message
        msg = await self.receive(timeout=10)

        if not msg:
            return

        # Check if we've already processed this message
        if msg.id in self._processed_messages:
            logger.debug(f"Skipping already processed message {msg.id}")
            return

        # Mark message as processed
        self._processed_messages.add(msg.id)
        logger.debug(f"LLMBehaviour received message: {msg}")

        conversation_id = generate_conversation_id(msg)

        # Initialize or retrieve conversation state
        if conversation_id not in self._active_conversations:
            self._active_conversations[conversation_id] = {
                "state": ConversationState.ACTIVE,
                "interaction_count": 0,
                "start_time": time.time(),
                "last_activity": time.time(),
            }

        conversation = self._active_conversations[conversation_id]

        # Check if conversation should be active
        if conversation["state"] != ConversationState.ACTIVE:
            logger.info(
                f"Conversation {conversation_id} is already in state {conversation['state']}, not processing further"
            )
            return

        # Update conversation tracking
        conversation["interaction_count"] += 1
        conversation["last_activity"] = time.time()

        # Check for max interactions
        if (
            self.max_interactions_per_conversation
            and conversation["interaction_count"] > self.max_interactions_per_conversation
        ):
            await self._end_conversation(
                conversation_id, ConversationState.MAX_INTERACTIONS_REACHED
            )

            reply = msg.make_reply()
            reply.body = f"This conversation has reached the maximum limit of {self.max_interactions_per_conversation} interactions."
            await self.send(reply)
            return

        # Apply input guardrails
        processed_content = await apply_input_guardrails(
            content=msg.body,
            message=msg,
            guardrails=self.input_guardrails,
            on_trigger=self.on_guardrail_trigger,
            send_reply=self.send,
        )
        if processed_content is None:
            # Message was blocked - the guardrail already sent a response
            return

        # Create a copy of the message with processed content
        processed_msg = Message(
            to=str(msg.to), sender=str(msg.sender), thread=msg.thread
        )
        processed_msg.body = processed_content

        # Update context with processed message
        self.context.add_message(processed_msg, conversation_id)

        # Auto-inject interaction memory if available
        await self._inject_interaction_memory(conversation_id)

        # Process the conversation with the LLM
        try:
            await self._process_message_with_llm(processed_msg, conversation_id)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._end_conversation(conversation_id, ConversationState.ERROR)

            # Send error message
            reply = msg.make_reply()
            reply.body = f"Error processing your message: {str(e)}"
            await self.send(reply)

    async def _process_message_with_llm(self, msg: Message, conversation_id: str):
        """
        Process a message with the LLM, handling multiple sequential tool calls.

        Args:
            msg: The message to process
            conversation_id: The ID of the conversation
        """
        final_response = None
        max_tool_iterations = (
            20  # Limit to prevent infinite loops -- should be parametrized
        )
        current_iteration = 0

        try:
            # Prepare tools with conversation context
            prepared_tools = self._prepare_tools_with_conversation_context(
                conversation_id
            )

            # Tool processing loop until a final response is obtained
            while final_response is None and current_iteration < max_tool_iterations:
                current_iteration += 1
                logger.info(
                    f"Tool processing iteration {current_iteration}/{max_tool_iterations}"
                )

                # Pass prepared tools to provider for this specific call
                llm_response = await self.provider.get_llm_response(
                    self.context, prepared_tools, conversation_id
                )

                tool_calls = llm_response.get("tool_calls", [])
                text_response = llm_response.get("text")

                if not tool_calls:
                    final_response = text_response
                    logger.info(
                        f"LLM provided final response without tools: {final_response[:100] if final_response else '(empty)'}..."
                    )
                    break

                logger.info(
                    f"LLM requested {len(tool_calls)} tool calls in iteration {current_iteration}"
                )

                # Use our helper function to create the assistant message with tool calls
                assistant_msg = create_assistant_tool_call_message(tool_calls)

                # Add the formatted message to context
                self.context.add_message_dict(assistant_msg, conversation_id)

                # Process each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})
                    tool_id = tool_call.get(
                        "id", f"call_{tool_name}_{current_iteration}"
                    )

                    logger.info(
                        f"Processing tool call: {tool_name} with args: {tool_args}"
                    )

                    # Find the tool by name in the prepared tools
                    tool = next(
                        (t for t in prepared_tools if t.name == tool_name), None
                    )

                    if tool:
                        try:
                            result = await tool.execute(**tool_args)

                            self.context.add_tool_result(
                                tool_name, result, tool_id, conversation_id
                            )

                            logger.info(f"Tool {tool_name} executed successfully")
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            logger.error(error_msg)
                            self.context.add_tool_result(
                                tool_name,
                                {"error": error_msg},
                                tool_id,
                                conversation_id,
                            )
                    else:
                        error_msg = f"Tool {tool_name} not found"
                        logger.error(error_msg)
                        self.context.add_tool_result(
                            tool_name, {"error": error_msg}, tool_id, conversation_id
                        )

            # Handle case where max iterations was reached
            if final_response is None and current_iteration >= max_tool_iterations:
                logger.warning(
                    f"Reached maximum tool iterations ({max_tool_iterations}), forcing final response"
                )
                final_response = (
                    await self.provider.get_llm_response(self.context, prepared_tools, conversation_id)
                ).get("text")

        except Exception as e:
            logger.error(f"Error in tool processing loop: {e}")
            # Instead of setting generic error message, re-raise to see actual error
            raise

        if not final_response:
            final_response = "I'm sorry, I couldn't complete this request properly. Please try again or rephrase your query."

        # Apply output guardrails before sending
        final_response = await apply_output_guardrails(
            content=final_response,
            original_message=msg,
            guardrails=self.output_guardrails,
            on_trigger=self.on_guardrail_trigger,
        )

        # Add assistant response to context before sending
        self.context.add_assistant_message(final_response, conversation_id)

        # Check for termination markers
        if final_response and any(
            marker in final_response for marker in self.termination_markers
        ):
            await self._end_conversation(conversation_id, ConversationState.COMPLETED)

        await self._send_response(final_response, msg, conversation_id)

    async def _send_response(
        self, response: str, original_msg: Message, conversation_id: str
    ) -> None:
        """
        Send response with optional conditional routing.

        Args:
            response: The LLM's response text
            original_msg: The original message received
            conversation_id: The conversation identifier
        """
        context = {
            "conversation_id": conversation_id,
            "state": self._active_conversations.get(conversation_id, {}),
        }

        # Determine recipients and transformations
        if self.routing_function:
            routing_result = self.routing_function(original_msg, response, context)

            if isinstance(routing_result, str):
                # Simple string routing
                recipients = [routing_result]
                transform = None
                metadata = {}
            elif isinstance(routing_result, RoutingResponse):
                # Advanced routing with RoutingResponse
                recipients = (
                    routing_result.recipients
                    if isinstance(routing_result.recipients, list)
                    else [routing_result.recipients]
                )
                transform = routing_result.transform
                metadata = routing_result.metadata or {}
            else:
                # Fallback to default behavior
                recipients = [self.reply_to or str(original_msg.sender)]
                transform = None
                metadata = {}
        else:
            # Traditional behavior when no routing function
            recipients = [self.reply_to or str(original_msg.sender)]
            transform = None
            metadata = {}

        # Send message to each recipient
        for recipient in recipients:
            reply = Message(to=recipient)
            reply.body = transform(response) if transform else response
            reply.thread = conversation_id

            # Mark as LLM message for proper template filtering
            reply.set_metadata("message_type", "llm")

            # Add any metadata
            for key, value in metadata.items():
                reply.set_metadata(key, value)

            logger.info(f"Sending response to {reply.to} with thread: {reply.thread}")

            try:
                await self.send(reply)
                logger.info(f"Response sent successfully to {reply.to}")
            except Exception as e:
                logger.error(f"Error sending response to {recipient}: {e}")

    async def _end_conversation(self, conversation_id: str, reason: str) -> None:
        """
        End a conversation and perform cleanup.

        Args:
            conversation_id: The ID of the conversation to end
            reason: The reason for ending the conversation
        """
        if conversation_id in self._active_conversations:
            self._active_conversations[conversation_id]["state"] = reason

            # Call the callback if provided
            if self.on_conversation_end:
                self.on_conversation_end(conversation_id, reason)

            logger.info(f"Conversation {conversation_id} ended: {reason}")

    def reset_conversation(self, conversation_id: str) -> bool:
        """
        Reset a conversation to allow it to continue beyond its limits.

        Args:
            conversation_id: The ID of the conversation to reset

        Returns:
            bool: True if the conversation was reset, False if not found
        """
        if conversation_id in self._active_conversations:
            self._active_conversations[conversation_id] = {
                "state": ConversationState.ACTIVE,
                "interaction_count": 0,
                "start_time": time.time(),
                "last_activity": time.time(),
            }
            return True
        return False

    def get_conversation_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a conversation.

        Args:
            conversation_id: The ID of the conversation

        Returns:
            Dict or None: The conversation state if found, None otherwise
        """
        return self._active_conversations.get(conversation_id)

    def register_tool(self, tool: LLMTool) -> None:
        """
        Register a tool with this behaviour.

        Args:
            tool: The tool to register
        """
        self.tools.append(tool)
        logger.info(f"Registered tool '{tool.name}' with behaviour")

    def get_tools(self) -> List[LLMTool]:
        """
        Get the list of tools registered with this behaviour.

        Returns:
            List of tools available to this behaviour
        """
        return self.tools

    def add_input_guardrail(self, guardrail: InputGuardrail) -> None:
        """
        Add an input guardrail to this behaviour.

        Args:
            guardrail: The input guardrail to add
        """
        self.input_guardrails.append(guardrail)
        logger.info(f"Added input guardrail '{guardrail.name}' to behaviour")

    def add_output_guardrail(self, guardrail: OutputGuardrail) -> None:
        """
        Add an output guardrail to this behaviour.

        Args:
            guardrail: The output guardrail to add
        """
        self.output_guardrails.append(guardrail)
        logger.info(f"Added output guardrail '{guardrail.name}' to behaviour")

    async def _inject_interaction_memory(self, conversation_id: str):
        """
        Inject relevant interaction memory into the context automatically.

        This method checks if we have previous interaction memory for this conversation
        and adds it to the context as a system message so the agent is aware of
        previous learned information without needing to explicitly recall it.

        Args:
            conversation_id: The conversation ID to check for memory
        """
        if not self.interaction_memory:
            return

        # Get memory summary for this conversation
        memory_summary = self.interaction_memory.get_context_summary(conversation_id)

        if memory_summary:
            # Check if we've already injected memory for this conversation
            # to avoid duplicating it on every message
            conversation_history = self.context.get_conversation_history(
                conversation_id
            )

            # Check if any existing message contains this memory summary
            memory_already_injected = any(
                msg.get("role") == "system"
                and "Previous interaction notes:" in msg.get("content", "")
                for msg in conversation_history
            )

            if not memory_already_injected:
                # Inject memory as a system message
                from ..context._types import create_system_message

                memory_message = create_system_message(memory_summary)
                self.context.add_message_dict(memory_message, conversation_id)
                logger.info(
                    f"Injected interaction memory for conversation {conversation_id}"
                )

    def _prepare_tools_with_conversation_context(self, conversation_id: str):
        """
        Prepare tools with conversation context for execution.

        This ensures that memory tools have access to the current conversation_id
        so they can store information correctly.

        Args:
            conversation_id: The current conversation ID

        Returns:
            List of tools prepared with conversation context
        """
        prepared_tools = []

        for tool in self.tools:
            # Check if this is a memory tool that needs conversation context
            if hasattr(tool, "set_conversation_id"):
                tool.set_conversation_id(conversation_id)
            prepared_tools.append(tool)

        return prepared_tools
