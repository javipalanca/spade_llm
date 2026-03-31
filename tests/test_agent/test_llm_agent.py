"""Tests for LLMAgent class."""

from unittest.mock import Mock, patch

import pytest

from spade_llm.agent import LLMAgent
from spade_llm.behaviour import LLMBehaviour
from spade_llm.context import ContextManager
from spade_llm.mcp import StdioServerConfig


class TestLLMAgentInitialization:
    """Test LLMAgent initialization."""

    def test_init_minimal(self, mock_llm_provider):
        """Test initialization with minimal parameters."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        assert agent.jid == "test@localhost"
        assert agent.password == "password"
        assert agent.provider == mock_llm_provider
        assert isinstance(agent.context, ContextManager)
        assert agent.reply_to is None
        assert agent.routing_function is None
        assert agent.tools == []
        assert agent.mcp_servers == []
        assert agent.termination_markers == ["<TASK_COMPLETE>", "<END>", "<DONE>"]
        assert agent.max_interactions_per_conversation is None
        assert agent.on_conversation_end is None
        assert isinstance(agent.llm_behaviour, LLMBehaviour)

    def test_init_full_parameters(self, mock_llm_provider, mock_simple_tool):
        """Test initialization with all parameters."""

        def routing_func(msg, response, context):
            return "target@localhost"

        def end_callback(conv_id, reason):
            pass

        mcp_server = StdioServerConfig(name="test_server", command="python", args=["test.py"])

        agent = LLMAgent(
            jid="test@localhost",
            password="password",
            provider=mock_llm_provider,
            reply_to="reply@localhost",
            routing_function=routing_func,
            system_prompt="Test system prompt",
            mcp_servers=[mcp_server],
            tools=[mock_simple_tool],
            termination_markers=["<STOP>"],
            max_interactions_per_conversation=5,
            on_conversation_end=end_callback,
            verify_security=True,
        )

        assert agent.reply_to == "reply@localhost"
        assert agent.routing_function == routing_func
        assert len(agent.tools) == 1
        assert agent.tools[0] == mock_simple_tool
        assert len(agent.mcp_servers) == 1
        assert agent.mcp_servers[0] == mcp_server
        assert agent.termination_markers == ["<STOP>"]
        assert agent.max_interactions_per_conversation == 5
        assert agent.on_conversation_end == end_callback
        assert agent.verify_security is True

    def test_llm_behaviour_creation(self, mock_llm_provider, mock_simple_tool):
        """Test that LLM behaviour is created correctly."""
        agent = LLMAgent(
            jid="test@localhost", password="password", provider=mock_llm_provider, tools=[mock_simple_tool]
        )

        behaviour = agent.llm_behaviour
        assert isinstance(behaviour, LLMBehaviour)
        assert behaviour.provider == mock_llm_provider
        assert behaviour.context == agent.context
        assert len(behaviour.tools) == 1
        assert behaviour.tools[0] == mock_simple_tool


class TestLLMAgentSetup:
    """Test LLMAgent setup functionality."""

    @pytest.mark.asyncio
    async def test_setup_without_mcp(self, mock_llm_provider):
        """Test setup without MCP servers."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        with patch.object(agent, "add_behaviour") as mock_add_behaviour:
            await agent.setup()

            # Should add the LLM behaviour
            mock_add_behaviour.assert_called_once()
            args = mock_add_behaviour.call_args[0]
            assert args[0] == agent.llm_behaviour

    @pytest.mark.asyncio
    async def test_setup_with_mcp_success(self, mock_llm_provider):
        """Test setup with MCP servers (success case)."""
        mcp_server = StdioServerConfig(name="test_server", command="python", args=["test.py"])

        agent = LLMAgent(
            jid="test@localhost", password="password", provider=mock_llm_provider, mcp_servers=[mcp_server]
        )

        mock_mcp_tools = [Mock(name="mcp_tool_1"), Mock(name="mcp_tool_2")]

        with patch("spade_llm.agent.llm_agent.get_all_mcp_tools", return_value=mock_mcp_tools) as mock_get_tools:
            with patch.object(agent, "add_behaviour") as mock_add_behaviour:
                await agent.setup()

                # Should fetch MCP tools
                mock_get_tools.assert_called_once_with([mcp_server])

                # Should add tools to agent
                assert len(agent.tools) == 2
                assert agent.tools == mock_mcp_tools

                # Should add the LLM behaviour
                mock_add_behaviour.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_with_mcp_error(self, mock_llm_provider):
        """Test setup with MCP servers (error case)."""
        mcp_server = StdioServerConfig(name="test_server", command="python", args=["test.py"])

        agent = LLMAgent(
            jid="test@localhost", password="password", provider=mock_llm_provider, mcp_servers=[mcp_server]
        )

        with patch("spade_llm.agent.llm_agent.get_all_mcp_tools", side_effect=Exception("MCP error")):
            with patch.object(agent, "add_behaviour") as mock_add_behaviour:
                # Should not raise exception, just log error
                await agent.setup()

                # Should still add the LLM behaviour
                mock_add_behaviour.assert_called_once()

                # Tools should remain empty
                assert len(agent.tools) == 0


class TestLLMAgentToolManagement:
    """Test tool management functionality."""

    def test_add_tool(self, mock_llm_provider, mock_simple_tool):
        """Test adding a tool to the agent."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        assert len(agent.tools) == 0

        # Mock the behaviour's register_tool method
        agent.llm_behaviour.register_tool = Mock()

        agent.add_tool(mock_simple_tool)

        assert len(agent.tools) == 1
        assert agent.tools[0] == mock_simple_tool

        # Should also register with behaviour
        agent.llm_behaviour.register_tool.assert_called_once_with(mock_simple_tool)

    def test_get_tools(self, mock_llm_provider, mock_simple_tool, mock_async_tool):
        """Test getting tools from the agent."""
        agent = LLMAgent(
            jid="test@localhost",
            password="password",
            provider=mock_llm_provider,
            tools=[mock_simple_tool, mock_async_tool],
        )

        tools = agent.get_tools()

        assert len(tools) == 2
        assert mock_simple_tool in tools
        assert mock_async_tool in tools


class TestLLMAgentConversationManagement:
    """Test conversation management functionality."""

    def test_reset_conversation(self, mock_llm_provider):
        """Test resetting a conversation."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        # Mock the behaviour's reset_conversation method
        agent.llm_behaviour.reset_conversation = Mock(return_value=True)

        result = agent.reset_conversation("test_conversation")

        assert result is True
        agent.llm_behaviour.reset_conversation.assert_called_once_with("test_conversation")

    def test_get_conversation_state(self, mock_llm_provider):
        """Test getting conversation state."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        expected_state = {"state": "active", "interaction_count": 3}
        agent.llm_behaviour.get_conversation_state = Mock(return_value=expected_state)

        result = agent.get_conversation_state("test_conversation")

        assert result == expected_state
        agent.llm_behaviour.get_conversation_state.assert_called_once_with("test_conversation")


class TestLLMAgentIntegration:
    """Test integration aspects of LLMAgent."""

    def test_behaviour_configuration_propagation(self, mock_llm_provider, mock_simple_tool):
        """Test that configuration is properly passed to the behaviour."""

        def routing_func(msg, response, context):
            return "target@localhost"

        def end_callback(conv_id, reason):
            pass

        agent = LLMAgent(
            jid="test@localhost",
            password="password",
            provider=mock_llm_provider,
            reply_to="reply@localhost",
            routing_function=routing_func,
            termination_markers=["<STOP>"],
            max_interactions_per_conversation=10,
            on_conversation_end=end_callback,
            tools=[mock_simple_tool],
        )

        behaviour = agent.llm_behaviour

        # Check that all parameters were passed to behaviour
        assert behaviour.provider == mock_llm_provider
        assert behaviour.reply_to == "reply@localhost"
        assert behaviour.routing_function == routing_func
        assert behaviour.context == agent.context
        assert behaviour.termination_markers == ["<STOP>"]
        assert behaviour.max_interactions_per_conversation == 10
        assert behaviour.on_conversation_end == end_callback
        assert len(behaviour.tools) == 1
        assert behaviour.tools[0] == mock_simple_tool

    def test_context_sharing(self, mock_llm_provider):
        """Test that agent and behaviour share the same context."""
        agent = LLMAgent(
            jid="test@localhost", password="password", provider=mock_llm_provider, system_prompt="Shared context test"
        )

        # Both should reference the same context instance
        assert agent.context is agent.llm_behaviour.context

    @pytest.mark.asyncio
    async def test_tool_registration_consistency(self, mock_llm_provider, mock_simple_tool):
        """Test that tools are consistently registered between agent and behaviour."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        # Mock the behaviour's register_tool method
        agent.llm_behaviour.register_tool = Mock()

        # Add tool to agent
        agent.add_tool(mock_simple_tool)

        # Check that tool is in both agent and behaviour
        assert mock_simple_tool in agent.tools
        agent.llm_behaviour.register_tool.assert_called_once_with(mock_simple_tool)

        # Check that get_tools returns the same tools
        assert agent.get_tools() == agent.tools


class TestLLMAgentErrorHandling:
    """Test error handling in LLMAgent."""

    def test_invalid_mcp_server_config(self, mock_llm_provider):
        """Test handling of invalid MCP server config."""
        # This should not raise during initialization
        agent = LLMAgent(
            jid="test@localhost",
            password="password",
            provider=mock_llm_provider,
            mcp_servers=[None],  # Invalid config
        )

        assert len(agent.mcp_servers) == 1
        assert agent.mcp_servers[0] is None

    def test_none_parameters(self, mock_llm_provider):
        """Test handling of None parameters."""
        agent = LLMAgent(
            jid="test@localhost",
            password="password",
            provider=mock_llm_provider,
            reply_to=None,
            routing_function=None,
            system_prompt=None,
            mcp_servers=None,
            tools=None,
            termination_markers=None,
            max_interactions_per_conversation=None,
            on_conversation_end=None,
        )

        # Should use defaults for None values
        assert agent.reply_to is None
        assert agent.routing_function is None
        assert agent.tools == []
        assert agent.mcp_servers == []
        assert agent.termination_markers == ["<TASK_COMPLETE>", "<END>", "<DONE>"]
        assert agent.max_interactions_per_conversation is None
        assert agent.on_conversation_end is None


class TestLLMAgentEdgeCases:
    """Test edge cases for LLMAgent."""

    def test_duplicate_tool_addition(self, mock_llm_provider, mock_simple_tool):
        """Test adding the same tool multiple times."""
        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider)

        # Mock the behaviour's register_tool method
        agent.llm_behaviour.register_tool = Mock()

        # Add same tool twice
        agent.add_tool(mock_simple_tool)
        agent.add_tool(mock_simple_tool)

        # Should have two references to the same tool
        assert len(agent.tools) == 2
        assert agent.tools[0] is mock_simple_tool
        assert agent.tools[1] is mock_simple_tool

        # Should call register_tool twice
        assert agent.llm_behaviour.register_tool.call_count == 2

    def test_very_long_jid(self, mock_llm_provider):
        """Test with very long JID."""
        long_jid = "a" * 1000 + "@localhost"

        agent = LLMAgent(jid=long_jid, password="password", provider=mock_llm_provider)

        assert agent.jid == long_jid

    def test_special_characters_in_password(self, mock_llm_provider):
        """Test with special characters in password."""
        special_password = "p@$$w0rd!@#$%^&*()_+-=[]{}|;':\",./<>?"

        agent = LLMAgent(jid="test@localhost", password=special_password, provider=mock_llm_provider)

        assert agent.password == special_password

    def test_large_number_of_tools(self, mock_llm_provider):
        """Test with a large number of tools."""
        tools = [Mock(name=f"tool_{i}") for i in range(100)]

        agent = LLMAgent(jid="test@localhost", password="password", provider=mock_llm_provider, tools=tools)

        assert len(agent.tools) == 100
        assert all(tool in agent.tools for tool in tools)
