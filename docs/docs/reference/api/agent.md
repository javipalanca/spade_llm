# Agent API

API reference for SPADE_LLM agent classes.

## LLMAgent

Main agent class that extends SPADE Agent with LLM capabilities.

### Constructor

```python
LLMAgent(
    jid: str,
    password: str,
    provider: LLMProvider,
    reply_to: Optional[str] = None,
    routing_function: Optional[RoutingFunction] = None,
    system_prompt: Optional[str] = None,
    mcp_servers: Optional[List[MCPServerConfig]] = None,
    tools: Optional[List[LLMTool]] = None,
    termination_markers: Optional[List[str]] = None,
    max_interactions_per_conversation: Optional[int] = None,
    on_conversation_end: Optional[Callable[[str, str], None]] = None,
    verify_security: bool = False
)
```

**Parameters:**

- `jid` - Jabber ID for the agent
- `password` - Agent password  
- `provider` - LLM provider instance
- `reply_to` - Optional fixed reply destination
- `routing_function` - Custom routing function
- `system_prompt` - System instructions for LLM
- `tools` - List of available tools
- `termination_markers` - Conversation end markers
- `max_interactions_per_conversation` - Conversation length limit
- `on_conversation_end` - Callback when conversation ends
- `verify_security` - Enable SSL verification

### Methods

#### add_tool(tool: LLMTool)

Add a tool to the agent.

```python
tool = LLMTool(name="function", description="desc", parameters={}, func=my_func)
agent.add_tool(tool)
```

#### get_tools() -> List[LLMTool]

Get all registered tools.

```python
tools = agent.get_tools()
print(f"Agent has {len(tools)} tools")
```

#### reset_conversation(conversation_id: str) -> bool

Reset conversation limits.

```python
success = agent.reset_conversation("user1_session")
```

#### get_conversation_state(conversation_id: str) -> Optional[Dict[str, Any]]

Get conversation state information.

```python
state = agent.get_conversation_state("user1_session")
if state:
    print(f"Interactions: {state['interaction_count']}")
```

### Example

```python
from spade_llm import LLMAgent, LLMProvider

provider = LLMProvider.create_openai(api_key="key", model="gpt-4o-mini")

agent = LLMAgent(
    jid="assistant@example.com",
    password="password",
    provider=provider,
    system_prompt="You are a helpful assistant",
    max_interactions_per_conversation=10
)

await agent.start()
```

## ChatAgent

Interactive chat agent for human-computer communication.

### Constructor

```python
ChatAgent(
    jid: str,
    password: str,
    target_agent_jid: str,
    display_callback: Optional[Callable[[str, str], None]] = None,
    on_message_sent: Optional[Callable[[str, str], None]] = None,
    on_message_received: Optional[Callable[[str, str], None]] = None,
    verbose: bool = False,
    verify_security: bool = False
)
```

**Parameters:**

- `target_agent_jid` - JID of agent to communicate with
- `display_callback` - Custom response display function  
- `on_message_sent` - Callback after sending message
- `on_message_received` - Callback after receiving response
- `verbose` - Enable detailed logging

### Methods

#### send_message(message: str)

Send message to target agent.

```python
chat_agent.send_message("Hello, how are you?")
```

#### send_message_async(message: str)

Send message asynchronously.

```python
await chat_agent.send_message_async("Hello!")
```

#### wait_for_response(timeout: float = 10.0) -> bool

Wait for response from target agent.

```python
received = await chat_agent.wait_for_response(timeout=30.0)
```

#### run_interactive()

Start interactive chat session.

```python
await chat_agent.run_interactive()  # Starts interactive chat
```

### Example

```python
from spade_llm import ChatAgent

def display_response(message: str, sender: str):
    print(f"Response: {message}")

chat_agent = ChatAgent(
    jid="human@example.com",
    password="password",
    target_agent_jid="assistant@example.com",
    display_callback=display_response
)

await chat_agent.start()
await chat_agent.run_interactive()  # Interactive chat
await chat_agent.stop()
```

## CoordinatorAgent

Specialized agent for orchestrating multiple SPADE subagents through a single LLM-driven coordinator.

### Constructor

```python
CoordinatorAgent(
    jid: str,
    password: str,
    subagent_ids: List[str],
    coordination_session: str = "main_coordination",
    provider: LLMProvider,
    routing_function: Optional[RoutingFunction] = None,
    **kwargs
)
```

**Parameters (in addition to `LLMAgent`):**

- `subagent_ids` — JIDs of the subagents managed by the coordinator (required).
- `coordination_session` — Thread identifier shared by all internal exchanges.
- `_response_timeout` *(attribute)* — Maximum time (seconds) the coordinator waits for a subagent reply (default `30.0`).

### Coordination Tools

Two tools are registered automatically during `setup()`:

| Tool | Purpose |
|------|---------|
| `send_to_agent(agent_id: str, command: str) -> str` | Sends a command to a registered subagent and waits for the reply within the coordination timeout. |
| `list_subagents() -> str` | Returns the current list of subagents and their last known status. |

### Behaviour

- All messages from or to managed subagents are forced into the shared `coordination_session`, giving the LLM full visibility of the organizational state.
- Custom routing ensures intermediate messages stay inside the organization, while termination markers (`<TASK_COMPLETE>`, `<END>`, `<DONE>`) trigger delivery to the original requester.
- Subagent status is tracked automatically (`unknown`, `sent_command`, `responded`, `timeout`), allowing the LLM to plan next steps.

### Example

```python
from spade_llm.agent import CoordinatorAgent
from spade_llm.providers import LLMProvider

subagents = [
    "traffic-analyzer@xmpp.local",
    "notification-service@xmpp.local",
]

coordinator = CoordinatorAgent(
    jid="city-coordinator@xmpp.local",
    password="secret",
    subagent_ids=subagents,
    provider=LLMProvider.create_openai(
        api_key="sk-...",
        model="gpt-4o-mini",
    ),
    coordination_session="city_ops"
)
await coordinator.start()
```

## Agent Lifecycle

### Starting Agents

```python
await agent.start()  # Initialize and connect
```

### Stopping Agents

```python
await agent.stop()  # Cleanup and disconnect
```

### Running with SPADE

```python
import spade

async def main():
    agent = LLMAgent(...)
    await agent.start()
    # Agent runs until stopped
    await agent.stop()

if __name__ == "__main__":
    spade.run(main())
```

## Error Handling

```python
try:
    await agent.start()
except ConnectionError:
    print("Failed to connect to XMPP server")
except ValueError:
    print("Invalid configuration")
```

## Best Practices

- Always call `start()` before using agents
- Use `stop()` for proper cleanup
- Handle connection errors gracefully
- Set appropriate conversation limits
- Use callbacks for monitoring
