"""Tests for ChatAgent class."""

import asyncio
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from spade_llm.agent import ChatAgent
from spade_llm.agent.chat_agent import run_interactive_chat, safe_input


class TestChatAgentInitialization:
    """Test ChatAgent initialization."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        assert agent.jid == "chat@localhost"
        assert agent.password == "password"
        assert agent.target_agent_jid == "target@localhost"
        assert agent.display_callback is None
        assert agent.on_message_sent is None
        assert agent.on_message_received is None
        assert agent.verbose is False
        assert agent.verify_security is False

    def test_init_full_parameters(self):
        """Test initialization with all parameters."""

        def display_callback(msg, sender):
            pass

        def on_sent(msg, recipient):
            pass

        def on_received(msg, sender):
            pass

        agent = ChatAgent(
            jid="chat@localhost",
            password="password",
            target_agent_jid="target@localhost",
            display_callback=display_callback,
            on_message_sent=on_sent,
            on_message_received=on_received,
            verbose=True,
            verify_security=True,
        )

        assert agent.display_callback == display_callback
        assert agent.on_message_sent == on_sent
        assert agent.on_message_received == on_received
        assert agent.verbose is True
        assert agent.verify_security is True


class TestChatAgentSetup:
    """Test ChatAgent setup functionality."""

    @pytest.mark.asyncio
    async def test_setup(self):
        """Test agent setup."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost", verbose=True)

        with patch.object(agent, "add_behaviour") as mock_add_behaviour:
            with patch.object(agent, "set") as mock_set:
                await agent.setup()

                # Should add two behaviours
                assert mock_add_behaviour.call_count == 2

                # Check that configuration was stored
                expected_calls = [
                    call("target_agent_jid", "target@localhost"),
                    call("message_to_send", None),
                    call("display_callback", None),
                    call("on_message_sent", None),
                    call("on_message_received", None),
                    call("verbose", True),
                    call("response_received", False),
                ]
                mock_set.assert_has_calls(expected_calls, any_order=True)


class TestChatAgentSendBehaviour:
    """Test ChatAgent SendBehaviour."""

    @pytest.mark.asyncio
    async def test_send_behaviour_no_message(self):
        """Test SendBehaviour when no message is queued."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        behaviour = agent.SendBehaviour()
        behaviour.get = Mock(return_value=None)
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should not send anything
        behaviour.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_behaviour_with_message(self):
        """Test SendBehaviour when message is queued."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        behaviour = agent.SendBehaviour()

        def mock_get(key):
            if key == "message_to_send":
                return "Test message"
            elif key == "target_agent_jid":
                return "target@localhost"
            elif key == "verbose":
                return False
            elif key == "on_message_sent":
                return None
            return None

        behaviour.get = Mock(side_effect=mock_get)
        behaviour.set = Mock()
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should send message
        behaviour.send.assert_called_once()
        sent_message = behaviour.send.call_args[0][0]
        assert sent_message.body == "Test message"
        assert sent_message.to == "target@localhost"

        # Should clear the queued message
        behaviour.set.assert_called_with("message_to_send", None)

    @pytest.mark.asyncio
    async def test_send_behaviour_with_callback(self):
        """Test SendBehaviour with callback."""
        callback = Mock()

        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        behaviour = agent.SendBehaviour()

        def mock_get(key):
            if key == "message_to_send":
                return "Test message"
            elif key == "target_agent_jid":
                return "target@localhost"
            elif key == "verbose":
                return False
            elif key == "on_message_sent":
                return callback
            return None

        behaviour.get = Mock(side_effect=mock_get)
        behaviour.set = Mock()
        behaviour.send = AsyncMock()

        await behaviour.run()

        # Should call callback
        callback.assert_called_once_with("Test message", "target@localhost")

    @pytest.mark.asyncio
    async def test_send_behaviour_verbose(self):
        """Test SendBehaviour in verbose mode."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost", verbose=True)

        behaviour = agent.SendBehaviour()

        def mock_get(key):
            if key == "message_to_send":
                return "Test message"
            elif key == "target_agent_jid":
                return "target@localhost"
            elif key == "verbose":
                return True
            elif key == "on_message_sent":
                return None
            return None

        behaviour.get = Mock(side_effect=mock_get)
        behaviour.set = Mock()
        behaviour.send = AsyncMock()

        with patch("spade_llm.agent.chat_agent.logger") as mock_logger:
            await behaviour.run()

            # Should log in verbose mode
            mock_logger.info.assert_called_once()


class TestChatAgentReceiveBehaviour:
    """Test ChatAgent ReceiveBehaviour."""

    @pytest.mark.asyncio
    async def test_receive_behaviour_no_message(self):
        """Test ReceiveBehaviour when no message is received."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        behaviour = agent.ReceiveBehaviour()
        behaviour.receive = AsyncMock(return_value=None)
        behaviour.get = Mock(return_value=None)
        behaviour.set = Mock()

        await behaviour.run()

        # Should not set response_received
        behaviour.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_receive_behaviour_with_message_default_display(self):
        """Test ReceiveBehaviour with message and default display."""
        mock_response = Mock()
        mock_response.body = "Response message"
        mock_response.sender = "sender@localhost"

        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        behaviour = agent.ReceiveBehaviour()
        behaviour.receive = AsyncMock(return_value=mock_response)

        def mock_get(key):
            if key == "display_callback":
                return None
            elif key == "on_message_received":
                return None
            return None

        behaviour.get = Mock(side_effect=mock_get)
        behaviour.set = Mock()

        with patch("builtins.print") as mock_print:
            await behaviour.run()

            # Should print with default format
            mock_print.assert_called_once_with("\nResponse from sender@localhost: 'Response message'")

            # Should mark response as received
            behaviour.set.assert_called_once_with("response_received", True)

    @pytest.mark.asyncio
    async def test_receive_behaviour_with_custom_display(self):
        """Test ReceiveBehaviour with custom display callback."""
        display_callback = Mock()
        on_received_callback = Mock()

        mock_response = Mock()
        mock_response.body = "Response message"
        mock_response.sender = "sender@localhost"

        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        behaviour = agent.ReceiveBehaviour()
        behaviour.receive = AsyncMock(return_value=mock_response)

        def mock_get(key):
            if key == "display_callback":
                return display_callback
            elif key == "on_message_received":
                return on_received_callback
            return None

        behaviour.get = Mock(side_effect=mock_get)
        behaviour.set = Mock()

        await behaviour.run()

        # Should call custom display callback
        display_callback.assert_called_once_with("Response message", "sender@localhost")

        # Should call received callback
        on_received_callback.assert_called_once_with("Response message", "sender@localhost")

        # Should mark response as received
        behaviour.set.assert_called_once_with("response_received", True)


class TestChatAgentMessageSending:
    """Test ChatAgent message sending methods."""

    def test_send_message(self):
        """Test send_message method."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        agent.set = Mock()

        agent.send_message("Test message")

        # Should queue message and reset response flag
        expected_calls = [call("message_to_send", "Test message"), call("response_received", False)]
        agent.set.assert_has_calls(expected_calls)

    @pytest.mark.asyncio
    async def test_send_message_async(self):
        """Test send_message_async method."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost", verbose=True)

        agent.send = AsyncMock()

        with patch("spade_llm.agent.chat_agent.logger") as mock_logger:
            await agent.send_message_async("Test message")

            # Should send message
            agent.send.assert_called_once()
            sent_message = agent.send.call_args[0][0]
            assert sent_message.body == "Test message"
            assert sent_message.to == "target@localhost"

            # Should log in verbose mode
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_async_with_callback(self):
        """Test send_message_async with callback."""
        callback = Mock()

        agent = ChatAgent(
            jid="chat@localhost", password="password", target_agent_jid="target@localhost", on_message_sent=callback
        )

        agent.send = AsyncMock()

        await agent.send_message_async("Test message")

        # Should call callback
        callback.assert_called_once_with("Test message", "target@localhost")


class TestChatAgentResponseWaiting:
    """Test ChatAgent response waiting functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_response_success(self):
        """Test wait_for_response when response is received."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        # Mock get to return True after a short delay
        call_count = 0

        def mock_get(key):
            nonlocal call_count
            call_count += 1
            if key == "response_received":
                return call_count > 2  # Return True after a few calls
            return None

        agent.get = Mock(side_effect=mock_get)

        result = await agent.wait_for_response(timeout=1.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_response_timeout(self):
        """Test wait_for_response when timeout occurs."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        # Mock get to always return False
        agent.get = Mock(return_value=False)

        result = await agent.wait_for_response(timeout=0.1)

        assert result is False

    def test_is_waiting_response(self):
        """Test is_waiting_response method."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        # Test when waiting
        agent.get = Mock(return_value=False)
        assert agent.is_waiting_response() is True

        # Test when not waiting
        agent.get = Mock(return_value=True)
        assert agent.is_waiting_response() is False


class TestChatAgentInteractiveMode:
    """Test ChatAgent interactive mode."""

    @pytest.mark.asyncio
    async def test_run_interactive(self):
        """Test run_interactive method."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        with patch("spade_llm.agent.chat_agent.run_interactive_chat") as mock_run:
            await agent.run_interactive(input_prompt="Custom> ", exit_command="quit", response_timeout=15.0)

            mock_run.assert_called_once_with(agent, input_prompt="Custom> ", exit_command="quit", response_timeout=15.0)


class TestSafeInputFunction:
    """Test safe_input utility function."""

    @pytest.mark.asyncio
    async def test_safe_input(self):
        """Test safe_input function."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = "user input"

            result = await safe_input("Enter text: ")

            assert result == "user input"
            mock_to_thread.assert_called_once_with(input, "Enter text: ")

    @pytest.mark.asyncio
    async def test_safe_input_default_prompt(self):
        """Test safe_input with default prompt."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = "user input"

            result = await safe_input()

            assert result == "user input"
            mock_to_thread.assert_called_once_with(input, "")


class TestRunInteractiveChatFunction:
    """Test run_interactive_chat utility function."""

    @pytest.mark.asyncio
    async def test_run_interactive_chat_exit_command(self):
        """Test run_interactive_chat with exit command."""
        mock_agent = Mock()
        mock_agent.send_message = Mock()
        mock_agent.wait_for_response = AsyncMock(return_value=True)

        with patch("spade_llm.agent.chat_agent.safe_input") as mock_input:
            with patch("builtins.print") as mock_print:
                # First call returns exit command
                mock_input.return_value = "exit"

                await run_interactive_chat(mock_agent)

                # Should print start message and end message
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Chat session started" in call for call in print_calls)
                assert any("Chat session ended" in call for call in print_calls)

                # Should not send any messages
                mock_agent.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_interactive_chat_normal_flow(self):
        """Test run_interactive_chat normal message flow."""
        mock_agent = Mock()
        mock_agent.send_message = Mock()
        mock_agent.wait_for_response = AsyncMock(return_value=True)

        with patch("spade_llm.agent.chat_agent.safe_input") as mock_input:
            with patch("builtins.print"):
                # First call returns message, second returns exit
                mock_input.side_effect = ["Hello", "exit"]

                await run_interactive_chat(mock_agent)

                # Should send the message
                mock_agent.send_message.assert_called_once_with("Hello")
                # Should wait for response
                mock_agent.wait_for_response.assert_called_once_with(10.0)

    @pytest.mark.asyncio
    async def test_run_interactive_chat_timeout(self):
        """Test run_interactive_chat with response timeout."""
        mock_agent = Mock()
        mock_agent.send_message = Mock()
        mock_agent.wait_for_response = AsyncMock(return_value=False)  # Timeout

        with patch("spade_llm.agent.chat_agent.safe_input") as mock_input:
            with patch("builtins.print") as mock_print:
                # First call returns message, second returns exit
                mock_input.side_effect = ["Hello", "exit"]

                await run_interactive_chat(mock_agent, response_timeout=5.0)

                # Should print timeout message
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Timeout waiting for response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_run_interactive_chat_empty_message(self):
        """Test run_interactive_chat with empty message."""
        mock_agent = Mock()
        mock_agent.send_message = Mock()
        mock_agent.wait_for_response = AsyncMock(return_value=True)

        with patch("spade_llm.agent.chat_agent.safe_input") as mock_input:
            with patch("builtins.print"):
                # First call returns empty string, second returns exit
                mock_input.side_effect = ["", "exit"]

                await run_interactive_chat(mock_agent)

                # Should not send empty message
                mock_agent.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_interactive_chat_keyboard_interrupt(self):
        """Test run_interactive_chat with KeyboardInterrupt."""
        mock_agent = Mock()

        with patch("spade_llm.agent.chat_agent.safe_input") as mock_input:
            with patch("builtins.print") as mock_print:
                # Raise KeyboardInterrupt
                mock_input.side_effect = KeyboardInterrupt()

                await run_interactive_chat(mock_agent)

                # Should print interrupt message
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Chat interrupted by user" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_run_interactive_chat_exception_handling(self):
        """Test run_interactive_chat exception handling."""
        mock_agent = Mock()
        mock_agent.send_message = Mock(side_effect=Exception("Test error"))

        with patch("spade_llm.agent.chat_agent.safe_input") as mock_input:
            with patch("builtins.print") as mock_print:
                # First call raises exception, second returns exit
                mock_input.side_effect = ["Hello", "exit"]

                await run_interactive_chat(mock_agent)

                # Should print error message and continue
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Error in chat loop" in call for call in print_calls)


class TestChatAgentEdgeCases:
    """Test edge cases for ChatAgent."""

    def test_very_long_target_jid(self):
        """Test with very long target JID."""
        long_jid = "a" * 1000 + "@localhost"

        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid=long_jid)

        assert agent.target_agent_jid == long_jid

    def test_all_callbacks_none(self):
        """Test with all callbacks set to None."""
        agent = ChatAgent(
            jid="chat@localhost",
            password="password",
            target_agent_jid="target@localhost",
            display_callback=None,
            on_message_sent=None,
            on_message_received=None,
        )

        assert agent.display_callback is None
        assert agent.on_message_sent is None
        assert agent.on_message_received is None

    @pytest.mark.asyncio
    async def test_concurrent_message_sending(self):
        """Test sending multiple messages concurrently."""
        agent = ChatAgent(jid="chat@localhost", password="password", target_agent_jid="target@localhost")

        agent.send = AsyncMock()

        # Send multiple messages concurrently
        tasks = [agent.send_message_async(f"Message {i}") for i in range(5)]

        await asyncio.gather(*tasks)

        # Should have sent all messages
        assert agent.send.call_count == 5
