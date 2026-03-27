# Routing API

API reference for message routing system.

## RoutingFunction

Type definition for routing functions.

```python
RoutingFunction = Callable[[Message, str, Dict[str, Any]], Union[str, RoutingResponse]]
```

**Parameters:**

- `msg` - Original SPADE message
- `response` - LLM response text
- `context` - Conversation context and metadata

**Returns:**

- `str` - Single recipient JID
- `List[str]` - Multiple recipient JIDs  
- `RoutingResponse` - Advanced routing
- `None` - Send to original sender

### Basic Routing Function

```python
def simple_router(msg, response, context):
    """Route based on response content."""
    if "technical" in response.lower():
        return "tech-support@example.com"
    elif "billing" in response.lower():
        return "billing@example.com"
    else:
        return str(msg.sender)  # Reply to sender
```

### Context Information

```python
context = {
    "conversation_id": "user1_agent1",
    "state": {
        "state": "active",
        "interaction_count": 5,
        "start_time": 1642678800.0,
        "last_activity": 1642678900.0
    }
}
```

## RoutingResponse

Advanced routing with transformations and metadata.

### Constructor

```python
@dataclass
class RoutingResponse:
    recipients: Union[str, List[str]]
    transform: Optional[Callable[[str], str]] = None
    metadata: Optional[Dict[str, Any]] = None
```

**Parameters:**

- `recipients` - Destination JID(s)
- `transform` - Function to modify response before sending
- `metadata` - Additional message metadata

### Example

```python
from spade_llm.routing import RoutingResponse

def advanced_router(msg, response, context):
    """Advanced routing with transformation."""
    
    def add_signature(text):
        return f"{text}\n\n--\nProcessed by AI Assistant"
    
    if "urgent" in response.lower():
        return RoutingResponse(
            recipients="emergency@example.com",
            transform=add_signature,
            metadata={
                "priority": "high",
                "category": "urgent",
                "original_sender": str(msg.sender)
            }
        )
    
    return str(msg.sender)
```

## Routing Patterns

### Content-Based Routing

```python
def content_router(msg, response, context):
    """Route based on response keywords."""
    response_lower = response.lower()
    
    routing_map = {
        "tech-support@example.com": ["error", "bug", "technical", "debug"],
        "sales@example.com": ["price", "cost", "purchase", "buy"],
        "billing@example.com": ["payment", "invoice", "billing"],
        "urgent@example.com": ["urgent", "emergency", "critical"]
    }
    
    for recipient, keywords in routing_map.items():
        if any(keyword in response_lower for keyword in keywords):
            return recipient
    
    return "general@example.com"
```

### Sender-Based Routing

```python
def sender_router(msg, response, context):
    """Route based on message sender."""
    sender = str(msg.sender)
    sender_domain = sender.split('@')[1]
    
    # Internal vs external routing
    if sender_domain == "company.com":
        return "internal-support@example.com"
    else:
        return "external-support@example.com"
```

### Context-Aware Routing

```python
def context_router(msg, response, context):
    """Route based on conversation context."""
    state = context.get("state", {})
    interaction_count = state.get("interaction_count", 0)
    
    # Long conversations need escalation
    if interaction_count > 10:
        return RoutingResponse(
            recipients="escalation@example.com",
            metadata={
                "reason": "long_conversation",
                "interaction_count": interaction_count
            }
        )
    
    return "standard@example.com"
```

### Multi-Recipient Routing

```python
def broadcast_router(msg, response, context):
    """Route to multiple recipients."""
    recipients = ["primary@example.com"]
    
    # Add recipients based on content
    if "error" in response.lower():
        recipients.append("monitoring@example.com")
    
    if "sales" in response.lower():
        recipients.append("sales-team@example.com")
    
    return RoutingResponse(
        recipients=recipients,
        metadata={
            "broadcast": True,
            "primary": "primary@example.com"
        }
    )
```

## Best Practices

### Routing Design

- **Keep logic simple** - Complex routing is hard to debug
- **Use meaningful destinations** - Clear JID naming
- **Handle edge cases** - Provide fallback routing
- **Document routing rules** - Clear rule descriptions
- **Test thoroughly** - Test all routing paths

