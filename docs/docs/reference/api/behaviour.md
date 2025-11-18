# Behaviour API

API reference for SPADE_LLM behaviour classes.

## LLMBehaviour

Core behaviour that handles LLM interaction loop.

### Constructor

```python
LLMBehaviour(
    llm_provider: LLMProvider,
    reply_to: Optional[str] = None,
    routing_function: Optional[RoutingFunction] = None,
    context_manager: Optional[ContextManager] = None,
    termination_markers: Optional[List[str]] = None,
    max_interactions_per_conversation: Optional[int] = None,
    on_conversation_end: Optional[Callable[[str, str], None]] = None,
    tools: Optional[List[LLMTool]] = None
)
```

**Parameters:**

- `llm_provider` - LLM provider instance
- `reply_to` - Fixed reply destination (optional)
- `routing_function` - Custom routing function
- `context_manager` - Context manager instance
- `termination_markers` - Conversation end markers
- `max_interactions_per_conversation` - Interaction limit
- `on_conversation_end` - End callback
- `tools` - Available tools

### Methods

#### register_tool(tool: LLMTool)

Register a tool with the behaviour.

```python
tool = LLMTool(name="func", description="desc", parameters={}, func=my_func)
behaviour.register_tool(tool)
```

#### get_tools() -> List[LLMTool]

Get registered tools.

```python
tools = behaviour.get_tools()
```

#### reset_conversation(conversation_id: str) -> bool

Reset conversation state.

```python
success = behaviour.reset_conversation("user1_session")
```

#### get_conversation_state(conversation_id: str) -> Optional[Dict[str, Any]]

Get conversation state.

```python
state = behaviour.get_conversation_state("user1_session")
```

### Processing Loop

The behaviour automatically:

1. Receives XMPP messages
2. Updates conversation context
3. Calls LLM provider
4. Executes requested tools
5. Routes responses

### Conversation States

```python
class ConversationState:
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    MAX_INTERACTIONS_REACHED = "max_interactions_reached"
```

### Example

```python
from spade_llm.behaviour import LLMBehaviour
from spade_llm.context import ContextManager

context = ContextManager(system_prompt="You are helpful")
behaviour = LLMBehaviour(
    llm_provider=provider,
    context_manager=context,
    termination_markers=["DONE", "END"],
    max_interactions_per_conversation=10
)

# Used internally by LLMAgent
agent.add_behaviour(behaviour)
```

## Internal Architecture

### Message Processing Flow

```python
async def run(self):
    """Main processing loop."""
    msg = await self.receive(timeout=10)
    if not msg:
        return
    
    # Update context
    self.context.add_message(msg, conversation_id)
    
    # Process with LLM
    await self._process_message_with_llm(msg, conversation_id)
```

### Tool Execution Loop

```python
async def _process_message_with_llm(self, msg, conversation_id):
    """Process message with tool execution."""
    max_iterations = 20
    current_iteration = 0
    
    while current_iteration < max_iterations:
        response = await self.provider.get_llm_response(self.context, self.tools)
        tool_calls = response.get('tool_calls', [])
        
        if not tool_calls:
            # Final response
            break
            
        # Execute tools
        for tool_call in tool_calls:
            await self._execute_tool(tool_call)
        
        current_iteration += 1
```

### Conversation Lifecycle

```python
def _end_conversation(self, conversation_id: str, reason: str):
    """End conversation and cleanup."""
    self._active_conversations[conversation_id]["state"] = reason
    
    if self.on_conversation_end:
        self.on_conversation_end(conversation_id, reason)
```

## Advanced Usage

### Custom Behaviour

```python
class CustomLLMBehaviour(LLMBehaviour):
    async def run(self):
        """Custom processing logic."""
        # Pre-processing
        await self.custom_preprocessing()
        
        # Standard processing
        await super().run()
        
        # Post-processing
        await self.custom_postprocessing()
    
    async def custom_preprocessing(self):
        """Custom preprocessing."""
        pass
    
    async def custom_postprocessing(self):
        """Custom postprocessing."""
        pass
```

### Direct Usage

```python
# Rarely used directly - usually through LLMAgent
from spade.agent import Agent
from spade.template import Template

class MyAgent(Agent):
    async def setup(self):
        behaviour = LLMBehaviour(llm_provider=provider)
        template = Template()
        self.add_behaviour(behaviour, template)
```

## Error Handling

The behaviour handles various error conditions:

- **Provider Errors**: LLM service failures
- **Tool Errors**: Tool execution failures  
- **Timeout Errors**: Response timeouts
- **Conversation Limits**: Max interaction limits

```python
try:
    await behaviour._process_message_with_llm(msg, conv_id)
except Exception as e:
    logger.error(f"Processing error: {e}")
    await behaviour._end_conversation(conv_id, ConversationState.ERROR)
```

## RetrievalBehaviour

Behaviour for handling document retrieval queries in [RetrievalAgent](agent.md#retrievalagent). This specialized behaviour extends `CyclicBehaviour` to continuously process incoming retrieval requests.

### Constructor

```python
RetrievalBehaviour(
    retriever: BaseRetriever,
    reply_to: Optional[str] = None,
    default_k: int = 4,
    on_retrieval_complete: Optional[Callable[[str, List[Any]], None]] = None
)
```

**Parameters:**
- `retriever` - Retriever instance for document retrieval
- `reply_to` - JID to send responses to. If None, replies to the original sender
- `default_k` - Default number of documents to retrieve (default: 4)
- `on_retrieval_complete` - Callback function when retrieval completes (receives query and results)

## Best Practices

- Let `LLMAgent` manage behaviour lifecycle
- Use appropriate termination markers
- Set reasonable interaction limits
- Handle conversation end events
- Monitor conversation states
- Use appropriate k values based on use case
