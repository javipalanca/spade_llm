"""Factory classes for creating mock objects in SPADE_LLM tests."""

from unittest.mock import AsyncMock, Mock
from typing import Optional, List, Callable

from spade.agent import Agent
from spade_llm.agent import LLMAgent
from spade_llm.behaviour import LLMBehaviour
from spade_llm.providers.base_provider import BaseLLMProvider
from spade_llm.context import ContextManager
from spade_llm.tools import LLMTool


class MockedConnectedAgent(Agent):
    """Mock base agent that simulates connection without real XMPP."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_connect = AsyncMock()
        self._async_register = AsyncMock()
        self.stream = Mock()
        self.client = Mock()
        
        # Mock client methods commonly used in tests
        self.client.send = AsyncMock()
        self.client.send_presence = Mock()
        self.client.event = Mock()


class MockedLLMAgent(LLMAgent):
    """Mock LLM agent that extends LLMAgent with mocked XMPP functionality."""
    
    def __init__(self, *args, **kwargs):
        # Initialize parent first
        super().__init__(*args, **kwargs)
        
        # Then mock the connection parts
        self._async_connect = AsyncMock()
        self._async_register = AsyncMock()
        self.stream = Mock()
        self.client = Mock()
        
        # Mock client methods
        self.client.send = AsyncMock()
        self.client.send_presence = Mock()
        self.client.event = Mock()
        
        # Mock the behaviour's send method to avoid real XMPP
        if hasattr(self, 'llm_behaviour'):
            self.llm_behaviour.send = AsyncMock()


def create_mocked_agent(jid: str = "test@localhost", password: str = "test_password"):
    """Create a basic mocked agent."""
    return MockedConnectedAgent(jid, password)


def create_mocked_llm_agent(jid: str = "llm_test@localhost", 
                           password: str = "test_password",
                           provider: Optional[BaseLLMProvider] = None):
    """Create a mocked LLM agent with optional provider."""
    if provider is None:
        from tests.conftest import MockLLMProvider
        provider = MockLLMProvider()
    
    return MockedLLMAgent(jid, password, provider)


class MockLLMBehaviour(LLMBehaviour):
    """Mock LLM behaviour that doesn't require real XMPP connection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send = AsyncMock()
        self.receive = AsyncMock()
        
        # Track sent messages for testing
        self.sent_messages = []
        
        async def mock_send(msg):
            self.sent_messages.append(msg)
            return True
            
        self.send.side_effect = mock_send


def create_mock_tool_with_result(name: str, result: any, should_error: bool = False):
    """
    Create a mock tool that returns a specific result or raises an error.
    
    Args:
        name: Name of the tool
        result: Result to return when executed
        should_error: Whether the tool should raise an error
    
    Returns:
        LLMTool instance
    """
    def tool_func(**kwargs):
        if should_error:
            raise RuntimeError(f"Mock error from {name}")
        return result
    
    return LLMTool(
        name=name,
        description=f"Mock tool {name} for testing",
        parameters={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input parameter"
                }
            }
        },
        func=tool_func
    )


def create_mock_conversation_state(conversation_id: str, 
                                 interaction_count: int = 1,
                                 state: str = "active"):
    """
    Create a mock conversation state dictionary.
    
    Args:
        conversation_id: ID of the conversation
        interaction_count: Number of interactions
        state: State of the conversation
    
    Returns:
        Dictionary representing conversation state
    """
    import time
    
    return {
        conversation_id: {
            "state": state,
            "interaction_count": interaction_count,
            "start_time": time.time() - 100,  # Started 100 seconds ago
            "last_activity": time.time()
        }
    }


def create_routing_function(target_jid: Optional[str] = None, 
                          transform_func: Optional[Callable] = None):
    """
    Create a mock routing function for testing.
    
    Args:
        target_jid: JID to route messages to
        transform_func: Function to transform message content
    
    Returns:
        Routing function
    """
    def routing_func(original_msg, response, context):
        if target_jid:
            return target_jid
        return str(original_msg.sender)
    
    return routing_func


def create_multiple_messages(count: int, base_body: str = "Test message"):
    """
    Create multiple mock messages for testing.
    
    Args:
        count: Number of messages to create
        base_body: Base body text for messages
    
    Returns:
        List of mock Message objects
    """
    messages = []
    
    for i in range(count):
        msg = Mock()
        msg.body = f"{base_body} {i+1}"
        msg.sender = f"sender{i+1}@localhost"
        msg.to = "receiver@localhost"
        msg.thread = f"thread-{i+1}"
        msg.id = f"msg_{i+1}"
        msg.metadata = {}
        
        # Mock make_reply
        reply = Mock()
        reply.to = msg.sender
        reply.sender = msg.to
        reply.body = ""
        reply.thread = msg.thread
        reply.metadata = {}
        reply.set_metadata = Mock()
        msg.make_reply = Mock(return_value=reply)
        
        messages.append(msg)
    
    return messages


class MockContextManager(ContextManager):
    """Mock context manager with additional testing features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_message_calls = []
        self.get_prompt_calls = []
        
    def add_message(self, message, conversation_id):
        """Override to track calls."""
        self.add_message_calls.append((message, conversation_id))
        return super().add_message(message, conversation_id)
        
    def get_prompt(self, conversation_id=None):
        """Override to track calls."""
        self.get_prompt_calls.append(conversation_id)
        return super().get_prompt(conversation_id)


def create_tool_call_response(tool_name: str, 
                            tool_args: dict, 
                            call_id: str = "call_123"):
    """
    Create a mock tool call response structure.
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments for the tool call
        call_id: ID of the tool call
    
    Returns:
        Tool call dictionary structure
    """
    return {
        "id": call_id,
        "name": tool_name,
        "arguments": tool_args
    }


def create_llm_response_with_tools(tool_calls: List[dict], text: Optional[str] = None):
    """
    Create a mock LLM response with tool calls.
    
    Args:
        tool_calls: List of tool call dictionaries
        text: Optional text response
    
    Returns:
        LLM response dictionary
    """
    return {
        "text": text,
        "tool_calls": tool_calls
    }
