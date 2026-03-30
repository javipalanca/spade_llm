# Guardrails API

API reference for content filtering and safety controls.

## Base Classes

### Guardrail

Abstract base class for all guardrail implementations.

```python
class Guardrail(ABC):
    def __init__(self, name: str, enabled: bool = True, blocked_message: Optional[str] = None)
```

**Parameters:**

- `name` - Unique identifier for the guardrail
- `enabled` - Whether the guardrail is active (default: True)
- `blocked_message` - Custom message when content is blocked

**Methods:**

```python
async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult
```

Abstract method that must be implemented by all guardrails.

```python
async def __call__(self, content: str, context: Dict[str, Any]) -> GuardrailResult
```

Main execution method that handles enabled state and post-processing.

### GuardrailResult

Result object returned by guardrail checks.

```python
@dataclass
class GuardrailResult:
    action: GuardrailAction
    content: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_message: Optional[str] = None
```

**Fields:**

- `action` - Action to take (PASS, MODIFY, BLOCK, WARNING)
- `content` - Modified content (if action is MODIFY)
- `reason` - Explanation for the action taken
- `metadata` - Additional information about the result
- `custom_message` - Custom message to send when blocked

### GuardrailAction

Enumeration of possible guardrail actions.

```python
class GuardrailAction(Enum):
    PASS = "pass"          # Allow without changes
    MODIFY = "modify"      # Modify the content
    BLOCK = "block"        # Block completely
    WARNING = "warning"    # Allow with warning
```

## Guardrail Types

### InputGuardrail

Base class for guardrails that process incoming messages.

```python
class InputGuardrail(Guardrail):
    def __init__(self, name: str, enabled: bool = True, blocked_message: Optional[str] = None)
```

Default blocked message: "Your message was blocked by security filters."

### OutputGuardrail

Base class for guardrails that process LLM responses.

```python
class OutputGuardrail(Guardrail):
    def __init__(self, name: str, enabled: bool = True, blocked_message: Optional[str] = None)
```

Default blocked message: "I apologize, but I cannot provide that response."

## Implementations

### KeywordGuardrail

Filters content based on keyword matching.

```python
KeywordGuardrail(
    name: str,
    blocked_keywords: List[str],
    action: GuardrailAction = GuardrailAction.BLOCK,
    replacement: str = "[REDACTED]",
    case_sensitive: bool = False,
    enabled: bool = True,
    blocked_message: Optional[str] = None
)
```

**Parameters:**

- `blocked_keywords` - List of keywords to filter
- `action` - Action to take when keyword found (BLOCK or MODIFY)
- `replacement` - Text to replace keywords with (if action is MODIFY)
- `case_sensitive` - Whether matching is case sensitive

**Example:**

```python
from spade_llm.guardrails import KeywordGuardrail, GuardrailAction

# Block harmful content
security_filter = KeywordGuardrail(
    name="security",
    blocked_keywords=["hack", "exploit", "malware"],
    action=GuardrailAction.BLOCK
)

# Replace profanity
profanity_filter = KeywordGuardrail(
    name="profanity",
    blocked_keywords=["damn", "hell"],
    action=GuardrailAction.MODIFY,
    replacement="[CENSORED]"
)
```

### RegexGuardrail

Applies regex patterns for content detection and modification.

```python
RegexGuardrail(
    name: str,
    patterns: Dict[str, Union[str, GuardrailAction]],
    enabled: bool = True,
    blocked_message: Optional[str] = None
)
```

**Parameters:**

- `patterns` - Dictionary mapping regex patterns to replacement strings or actions

**Example:**

```python
from spade_llm.guardrails import RegexGuardrail, GuardrailAction

# Redact personal information
pii_filter = RegexGuardrail(
    name="pii_protection",
    patterns={
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
        r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b': GuardrailAction.BLOCK
    },
    blocked_message="Credit card information is not allowed."
)
```

### LLMGuardrail

Uses a separate LLM model to validate content safety.

```python
LLMGuardrail(
    name: str,
    provider: LLMProvider,
    safety_prompt: Optional[str] = None,
    enabled: bool = True,
    blocked_message: Optional[str] = None
)
```

**Parameters:**

- `provider` - LLM provider to use for safety validation
- `safety_prompt` - Custom prompt for safety checking

**Example:**

```python
from spade_llm.guardrails import LLMGuardrail
from spade_llm.providers import LLMProvider

safety_provider = LLMProvider(
    model="gpt-5-nano",
    api_key="your-key",
)

ai_safety = LLMGuardrail(
    name="ai_safety_check",
    provider=safety_provider,
    safety_prompt="""
    Analyze this text for harmful content.
    Respond with JSON: {"safe": true/false, "reason": "explanation"}
    
    Text: {content}
    """
)
```

### CustomFunctionGuardrail

Allows custom validation logic using user-defined functions.

```python
CustomFunctionGuardrail(
    name: str,
    check_function: Callable[[str, Dict[str, Any]], GuardrailResult],
    enabled: bool = True,
    blocked_message: Optional[str] = None
)
```

**Parameters:**

- `check_function` - Function that performs the validation

**Example:**

```python
from spade_llm.guardrails import CustomFunctionGuardrail, GuardrailResult, GuardrailAction

def length_check(content: str, context: dict) -> GuardrailResult:
    if len(content) > 1000:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason="Message too long"
        )
    return GuardrailResult(action=GuardrailAction.PASS, content=content)

length_filter = CustomFunctionGuardrail(
    name="length_limit",
    check_function=length_check
)
```

### CompositeGuardrail

Combines multiple guardrails into a processing pipeline.

```python
CompositeGuardrail(
    name: str,
    guardrails: List[Guardrail],
    stop_on_block: bool = True,
    enabled: bool = True,
    blocked_message: Optional[str] = None
)
```

**Parameters:**

- `guardrails` - List of guardrails to apply in sequence
- `stop_on_block` - Whether to stop processing on first block

**Example:**

```python
from spade_llm.guardrails import CompositeGuardrail

security_pipeline = CompositeGuardrail(
    name="security_pipeline",
    guardrails=[
        KeywordGuardrail("keywords", ["hack"], GuardrailAction.BLOCK),
        RegexGuardrail("pii", email_patterns),
        LLMGuardrail("ai_safety", safety_provider)
    ],
    stop_on_block=True
)
```

## Processing Functions

### apply_input_guardrails()

Processes input guardrails and returns filtered content.

```python
async def apply_input_guardrails(
    content: str,
    message: Message,
    guardrails: List[InputGuardrail],
    on_trigger: Optional[Callable[[GuardrailResult], None]] = None,
    send_reply: Optional[Callable[[Message], None]] = None
) -> Optional[str]
```

**Parameters:**

- `content` - Input content to process
- `message` - Original SPADE message
- `guardrails` - List of input guardrails to apply
- `on_trigger` - Callback for guardrail events
- `send_reply` - Function to send blocking responses

**Returns:**

- `str` - Processed content if passed guardrails
- `None` - If content was blocked

### apply_output_guardrails()

Processes output guardrails and returns filtered content.

```python
async def apply_output_guardrails(
    content: str,
    original_message: Message,
    guardrails: List[OutputGuardrail],
    on_trigger: Optional[Callable[[GuardrailResult], None]] = None
) -> str
```

**Parameters:**

- `content` - LLM response content to process
- `original_message` - Original input message
- `guardrails` - List of output guardrails to apply
- `on_trigger` - Callback for guardrail events

**Returns:**

- `str` - Processed content (never None - blocked content returns safe message)

## Context Information

Guardrails receive context information for decision making:

### Input Guardrail Context

```python
{
    "message": Message,           # Original SPADE message
    "sender": str,               # Message sender JID
    "conversation_id": str       # Conversation identifier
}
```

### Output Guardrail Context

```python
{
    "original_message": Message, # Original input message
    "conversation_id": str,      # Conversation identifier
    "llm_response": str         # Original LLM response
}
```

## Agent Integration

### Adding Guardrails to Agents

```python
from spade_llm import LLMAgent

agent = LLMAgent(
    jid="assistant@example.com",
    password="password",
    provider=provider,
    input_guardrails=[input_filter1, input_filter2],
    output_guardrails=[output_filter1, output_filter2],
    on_guardrail_trigger=callback_function
)
```

### Dynamic Guardrail Management

```python
# Add guardrails at runtime
agent.add_input_guardrail(new_filter)
agent.add_output_guardrail(safety_check)

# Control individual guardrails
agent.input_guardrails[0].enabled = False  # Disable specific guardrail
```

## Best Practices

### Performance Considerations

- **LLM-based guardrails** are slower - use sparingly
- **Keyword guardrails** are fastest for simple filtering
- **Regex guardrails** offer good performance/flexibility balance
- Consider **caching** for expensive operations

### Security Guidelines

- Use **multiple layers** of protection
- **Test thoroughly** with diverse inputs
- **Monitor performance** impact
- **Log all guardrail actions** for auditing
- **Fail safely** - prefer allowing content over breaking functionality

### Error Handling

```python
try:
    result = await guardrail.check(content, context)
except Exception as e:
    # Guardrails fail safely - log error and allow content
    logger.error(f"Guardrail {guardrail.name} failed: {e}")
    result = GuardrailResult(action=GuardrailAction.PASS, content=content)
```
