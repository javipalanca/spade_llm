# Context API

API reference for conversation context management and strategies.

## ContextManager

Manages conversation history and context for LLM interactions.

### Constructor

```python
ContextManager(
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None
)
```

**Parameters:**

- `max_tokens` - Maximum context size in tokens
- `system_prompt` - System instructions for LLM

### Methods

#### add_message()

```python
def add_message(self, message: Message, conversation_id: str) -> None
```

Add SPADE message to conversation context.

**Example:**

```python
context = ContextManager(system_prompt="You are helpful")
context.add_message(spade_message, "user1_session")
```

#### add_message_dict()

```python
def add_message_dict(self, message_dict: ContextMessage, conversation_id: str) -> None
```

Add message from dictionary format.

**Example:**

```python
user_msg = {"role": "user", "content": "Hello!"}
context.add_message_dict(user_msg, "user1_session")
```

#### add_assistant_message()

```python
def add_assistant_message(self, content: str, conversation_id: Optional[str] = None) -> None
```

Add assistant response to context.

**Example:**

```python
context.add_assistant_message("Hello! How can I help?", "user1_session")
```

#### add_tool_result()

```python
def add_tool_result(
    self, 
    tool_name: str, 
    result: Any, 
    tool_call_id: str, 
    conversation_id: Optional[str] = None
) -> None
```

Add tool execution result to context.

**Example:**

```python
context.add_tool_result(
    tool_name="get_weather",
    result="22°C, sunny",
    tool_call_id="call_123",
    conversation_id="user1_session"
)
```

#### get_prompt()

```python
def get_prompt(self, conversation_id: Optional[str] = None) -> List[ContextMessage]
```

Get formatted prompt for LLM provider.

**Example:**

```python
prompt = context.get_prompt("user1_session")
# Returns list of messages formatted for LLM
```

#### get_conversation_history()

```python
def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[ContextMessage]
```

Get raw conversation history.

**Example:**

```python
history = context.get_conversation_history("user1_session")
print(f"Conversation has {len(history)} messages")
```

#### clear()

```python
def clear(self, conversation_id: Optional[str] = None) -> None
```

Clear conversation messages.

**Example:**

```python
# Clear specific conversation
context.clear("user1_session")

# Clear all conversations
context.clear("all")
```

#### get_active_conversations()

```python
def get_active_conversations(self) -> List[str]
```

Get list of active conversation IDs.

**Example:**

```python
conversations = context.get_active_conversations()
print(f"Active conversations: {conversations}")
```

#### set_current_conversation()

```python
def set_current_conversation(self, conversation_id: str) -> bool
```

Set current conversation context.

**Example:**

```python
success = context.set_current_conversation("user1_session")
```

### Example Usage

```python
from spade_llm.context import ContextManager

# Create context manager
context = ContextManager(
    system_prompt="You are a helpful coding assistant",
    max_tokens=2000
)

# Add conversation messages
context.add_message_dict(
    {"role": "user", "content": "Help me with Python"}, 
    "coding_session"
)

context.add_assistant_message(
    "I'd be happy to help with Python!", 
    "coding_session"
)

# Get formatted prompt
prompt = context.get_prompt("coding_session")
# Use with LLM provider
```

## CoordinationContextManager

Context manager tailored for coordination scenarios where a coordinator agent supervises several subagents.

```python
CoordinationContextManager(
    coordination_session: str,
    subagent_ids: Set[str],
    **kwargs
)
```

**Additional behaviour:**

- Forces every message involving a registered subagent to use the shared `coordination_session` thread.
- Preserves standard handling for external participants (messages outside the organization keep their original thread).
- Provides `add_coordination_command(target_agent: str, command: str)` to record coordinator-issued instructions inside the shared context.

**Usage Example:**

```python
from spade_llm.agent.coordinator_agent import CoordinationContextManager

coordination_context = CoordinationContextManager(
    coordination_session="city_ops",
    subagent_ids={"traffic@xmpp.local", "alerts@xmpp.local"}
)

# Attach to CoordinatorAgent via `_context_override` or constructor kwargs
```

## Context Management Strategies

### ContextManagement (Abstract Base)

Base class for all context management strategies.

```python
from spade_llm.context.management import ContextManagement
```

#### apply_context_strategy()

```python
def apply_context_strategy(
    self, 
    messages: List[ContextMessage], 
    system_prompt: Optional[str] = None
) -> List[ContextMessage]
```

Apply the context management strategy to messages.

#### get_stats()

```python
def get_stats(self, total_messages: int) -> Dict[str, Any]
```

Get statistics about context management.

### NoContextManagement

Preserves all messages without any filtering.

```python
from spade_llm.context import NoContextManagement

context = NoContextManagement()
```

### WindowSizeContext

Basic sliding window context management.

```python
from spade_llm.context import WindowSizeContext

context = WindowSizeContext(max_messages=20)
```

**Parameters:**
- `max_messages` (int): Maximum number of messages to keep

### SmartWindowSizeContext

Context management with message selection.

```python
from spade_llm.context import SmartWindowSizeContext

context = SmartWindowSizeContext(
    max_messages=20,
    preserve_initial=3,
    prioritize_tools=True
)
```

**Parameters:**
- `max_messages` (int): Maximum number of messages to keep
- `preserve_initial` (int, optional): Number of initial messages to always preserve
- `prioritize_tools` (bool, optional): Whether to prioritize tool result messages

**Example:**

```python
# Smart context with tool prioritization
smart_context = SmartWindowSizeContext(
    max_messages=25,
    preserve_initial=3,
    prioritize_tools=True
)

# Get statistics
stats = smart_context.get_stats(total_messages=50)
# Returns: {"strategy": "smart_window_size", "max_messages": 25, ...}
```

## Message Types

### ContextMessage Types

```python
from spade_llm.context._types import (
    SystemMessage,
    UserMessage, 
    AssistantMessage,
    ToolResultMessage
)
```

#### SystemMessage

```python
{
    "role": "system",
    "content": "You are a helpful assistant"
}
```

#### UserMessage

```python
{
    "role": "user",
    "content": "Hello, how are you?",
    "name": "user@example.com"  # Optional
}
```

#### AssistantMessage

```python
# Text response
{
    "role": "assistant",
    "content": "I'm doing well, thank you!"
}

# With tool calls
{
    "role": "assistant", 
    "content": None,
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"Madrid\"}"
            }
        }
    ]
}
```

#### ToolResultMessage

```python
{
    "role": "tool",
    "content": "Weather in Madrid: 22°C, sunny",
    "tool_call_id": "call_123"
}
```


