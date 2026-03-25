"""LLM-capable agent implementation for SPADE."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from spade.agent import Agent
from spade.template import Template

from ..behaviour import LLMBehaviour
from ..context import ContextManager
from ..context.management import ContextManagement
from ..guardrails import GuardrailResult, InputGuardrail, OutputGuardrail
from ..mcp import MCPServerConfig, get_all_mcp_tools
from ..memory import AgentInteractionMemory, AgentMemoryTool
from ..providers.base_provider import BaseLLMProvider
from ..routing import RoutingFunction
from ..tools import LLMTool

logger = logging.getLogger("spade_llm.agent")


class LLMAgent(Agent):
    """
    A SPADE agent with integrated LLM capabilities.

    This agent extends the standard SPADE agent with:
    - Built-in LLM integration
    - Context management
    - Tool execution framework
    - MCP server support
    - Conditional routing based on LLM responses
    - Input and output guardrails for content filtering
    """

    def __init__(
        self,
        jid: str,
        password: str,
        provider: BaseLLMProvider,
        reply_to: Optional[str] = None,
        routing_function: Optional[RoutingFunction] = None,
        system_prompt: Optional[str] = None,
        context_management: Optional[ContextManagement] = None,
        mcp_servers: Optional[List[MCPServerConfig]] = None,
        tools: Optional[List[LLMTool]] = None,
        termination_markers: Optional[List[str]] = None,
        max_interactions_per_conversation: Optional[int] = None,
        on_conversation_end: Optional[Callable[[str, str], None]] = None,
        input_guardrails: Optional[List[InputGuardrail]] = None,
        output_guardrails: Optional[List[OutputGuardrail]] = None,
        on_guardrail_trigger: Optional[Callable[[GuardrailResult], None]] = None,
        interaction_memory: Union[bool, Tuple[bool, str]] = False,
        agent_base_memory: Union[bool, Tuple[bool, str]] = False,
        verify_security: bool = False,
        **kwargs
    ):
        """
        Initialize an LLM-capable agent.

        Args:
            jid: The Jabber ID of the agent
            password: The password for the agent
            provider: The LLM provider to use
            reply_to: JID to send responses to. If None the reply is to the original sender
            routing_function: Optional function for conditional routing based on response content
            system_prompt: Optional system instructions for the LLM
            context_management: Optional context management strategy for controlling conversation history
            mcp_servers: Optional list of MCP server configurations
            tools: Optional list of tools the agent can use
            termination_markers: List of strings that mark conversation completion
            max_interactions_per_conversation: Maximum number of back-and-forth exchanges in a conversation
            on_conversation_end: Callback function when a conversation ends (receives conversation_id and reason)
            input_guardrails: List of guardrails to apply to incoming messages
            output_guardrails: List of guardrails to apply to LLM responses
            on_guardrail_trigger: Callback when a guardrail blocks/modifies content
            interaction_memory: Enable interaction memory. Can be:
                - bool: True/False (uses default path)
                - tuple: (True, "/custom/path") for custom path
            agent_base_memory: Enable agent base memory. Can be:
                - bool: True/False (uses default path)
                - tuple: (True, "/custom/path") for custom path
            verify_security: Whether to verify security certificates
        """
        super().__init__(jid, password, verify_security=verify_security)

        # Allow subclasses to inject custom context managers
        if '_context_override' in kwargs:
            self.context = kwargs.pop('_context_override')
        else:
            self.context = ContextManager(
                system_prompt=system_prompt, context_management=context_management
            )
        self.provider = provider
        self.context_management = context_management
        self.reply_to = reply_to
        self.routing_function = routing_function
        self.tools: List[LLMTool] = []
        self.mcp_servers = mcp_servers or []

        # Process initial tools and bind HumanInTheLoopTools to this agent
        if tools:
            for tool in tools:
                self._register_tool(tool)

        # Setup interaction memory if enabled
        self.interaction_memory = None
        interaction_enabled, interaction_path = self._parse_memory_config(
            interaction_memory
        )
        if interaction_enabled:
            self.interaction_memory = AgentInteractionMemory(jid, interaction_path)
            memory_tool = AgentMemoryTool(self.interaction_memory)
            self._register_tool(memory_tool)
            logger.info(
                f"Enabled interaction memory for agent {jid} with path: {interaction_path or 'default'}"
            )

        # Setup agent base memory if enabled
        self.agent_base_memory = None
        base_memory_enabled, base_memory_path = self._parse_memory_config(
            agent_base_memory
        )
        if base_memory_enabled:
            from ..memory.agent_base_memory import AgentBaseMemory
            from ..memory.agent_base_memory_tools import create_base_memory_tools

            self.agent_base_memory = AgentBaseMemory(jid, memory_path=base_memory_path)
            base_memory_tools = create_base_memory_tools(self.agent_base_memory)

            for tool in base_memory_tools:
                self._register_tool(tool)

            logger.info(
                f"Enabled agent base memory for agent {jid} with path: {base_memory_path or 'default'}"
            )

        self.termination_markers = termination_markers or [
            "<TASK_COMPLETE>",
            "<END>",
            "<DONE>",
        ]
        self.max_interactions_per_conversation = max_interactions_per_conversation
        self.on_conversation_end = on_conversation_end

        # Guardrails
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.on_guardrail_trigger = on_guardrail_trigger

        # Create LLM behaviour with all parameters
        self.llm_behaviour = LLMBehaviour(
            llm_provider=provider,
            reply_to=self.reply_to,
            routing_function=self.routing_function,
            context_manager=self.context,
            termination_markers=self.termination_markers,
            max_interactions_per_conversation=self.max_interactions_per_conversation,
            on_conversation_end=self.on_conversation_end,
            tools=self.tools,
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails,
            on_guardrail_trigger=self.on_guardrail_trigger,
            interaction_memory=self.interaction_memory,
        )

    async def setup(self):
        """Set up the agent with LLM behaviour and MCP tools."""
        logger.info(f"LLMAgent starting: {self.jid}")

        # Register MCP tools if MCP servers are configured
        if self.mcp_servers:
            await self._setup_mcp_tools()

        # LLM-targeted messages only
        template = Template()
        template.set_metadata("message_type", "llm")
        self.add_behaviour(self.llm_behaviour, template)

    async def _setup_mcp_tools(self):
        """Set up tools from configured MCP servers."""
        try:
            # Get tools from all MCP servers
            mcp_tools = await get_all_mcp_tools(self.mcp_servers)

            # Add each tool to the agent's tools list
            for tool in mcp_tools:
                self.add_tool(tool)

            logger.info(
                f"Registered {len(mcp_tools)} MCP tools from {len(self.mcp_servers)} servers"
            )
        except Exception as e:
            logger.error(f"Error setting up MCP tools: {e}")

    def _parse_memory_config(
        self, memory_config: Union[bool, Tuple[bool, str]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Parse memory configuration parameter.

        Args:
            memory_config: Can be:
                - bool: True/False (uses default path)
                - tuple: (enabled, custom_path)

        Returns:
            Tuple of (enabled, path)
        """
        if isinstance(memory_config, bool):
            return memory_config, None
        elif isinstance(memory_config, tuple) and len(memory_config) == 2:
            enabled, path = memory_config
            if isinstance(enabled, bool) and isinstance(path, str):
                return enabled, path
            else:
                logger.warning(
                    f"Invalid memory config tuple format: {memory_config}. Using default."
                )
                return False, None
        else:
            logger.warning(
                f"Invalid memory config type: {type(memory_config)}. Expected bool or tuple."
            )
            return False, None

    def _register_tool(self, tool: LLMTool):
        """
        Internal method to register a tool and handle special setup.

        Args:
            tool: The tool to register
        """
        # Check if it's a HumanInTheLoopTool and bind it to this agent
        if hasattr(tool, "set_agent") and callable(getattr(tool, "set_agent")):
            tool.set_agent(self)

        self.tools.append(tool)

    def add_tool(self, tool: LLMTool):
        """
        Add a tool to the agent.

        Args:
            tool: The tool to add
        """
        self._register_tool(tool)
        # Also register with the behaviour
        self.llm_behaviour.register_tool(tool)

    def reset_conversation(self, conversation_id: str) -> bool:
        """
        Reset a conversation to allow it to continue beyond its limits.

        Args:
            conversation_id: The ID of the conversation to reset

        Returns:
            bool: True if the conversation was reset, False if not found
        """
        return self.llm_behaviour.reset_conversation(conversation_id)

    def get_conversation_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a conversation.

        Args:
            conversation_id: The ID of the conversation

        Returns:
            Dict or None: The conversation state if found, None otherwise
        """
        return self.llm_behaviour.get_conversation_state(conversation_id)

    def get_tools(self) -> List[LLMTool]:
        """
        Get the list of tools available to this agent.

        Returns:
            List of tools
        """
        return self.tools

    def add_input_guardrail(self, guardrail: InputGuardrail):
        """
        Add an input guardrail to the agent.

        Args:
            guardrail: The input guardrail to add
        """
        self.input_guardrails.append(guardrail)
        self.llm_behaviour.add_input_guardrail(guardrail)

    def add_output_guardrail(self, guardrail: OutputGuardrail):
        """
        Add an output guardrail to the agent.

        Args:
            guardrail: The output guardrail to add
        """
        self.output_guardrails.append(guardrail)
        self.llm_behaviour.add_output_guardrail(guardrail)

    def get_context_stats(
        self, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get context management statistics for a conversation.

        Args:
            conversation_id: Optional ID of the conversation to get stats for

        Returns:
            Dictionary with context management statistics
        """
        return self.context.get_context_stats(conversation_id)

    def update_context_management(
        self, new_context_management: ContextManagement
    ) -> None:
        """
        Update the context management strategy.

        Args:
            new_context_management: New context management strategy to use
        """
        self.context_management = new_context_management
        self.context.update_context_management(new_context_management)
