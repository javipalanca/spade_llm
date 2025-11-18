# Guardrails System Tutorial

In this tutorial, you'll learn how to implement safety and content filtering in your SPADE-LLM agents using the guardrails system. We'll build on the concepts from the [First Agent Tutorial](first-agent.md) and add layers of protection.

## What are Guardrails?

Guardrails are safety mechanisms that filter and validate content before it reaches your LLM or before responses are sent to users. They provide essential protection against:

- Harmful or malicious content
- Inappropriate language
- Sensitive information leakage
- Policy violations
- Unsafe AI responses

## Types of Guardrails

SPADE-LLM supports two types of guardrails:

1. **Input Guardrails**: Filter incoming messages before they reach the LLM
2. **Output Guardrails**: Validate LLM responses before sending them to users

## Prerequisites

- Complete the [First Agent Tutorial](first-agent.md)
- SPADE-LLM installed with all dependencies
- Access to an LLM provider (OpenAI or Ollama)
- XMPP server running

## Step 1: Basic Keyword Filtering

Let's start with simple keyword-based filtering. This example blocks harmful content:

```python
import spade
import getpass
import logging
from spade_llm import LLMAgent, ChatAgent, LLMProvider
from spade_llm.guardrails import KeywordGuardrail, GuardrailAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("🛡️ Guardrails Tutorial: Basic Keyword Filtering")
    
    # Configuration
    xmpp_server = input("XMPP server domain (default: localhost): ") or "localhost"
    
    # Create provider
    provider = LLMProvider.create_openai(
        api_key=getpass.getpass("OpenAI API key: "),
        model="gpt-4o-mini"
    )
    
    # Create keyword guardrail that BLOCKS harmful content
    safety_guardrail = KeywordGuardrail(
        name="harmful_content_filter",
        blocked_keywords=["hack", "exploit", "malware", "virus", "illegal", "bomb"],
        action=GuardrailAction.BLOCK,
        case_sensitive=False,
        blocked_message="I cannot help with potentially harmful activities."
    )
    
    # Create LLM agent with input guardrail
    llm_agent = LLMAgent(
        jid=f"safe_assistant@{xmpp_server}",
        password=getpass.getpass("LLM agent password: "),
        provider=provider,
        system_prompt="You are a helpful and safe AI assistant.",
        input_guardrails=[safety_guardrail]  # Apply to incoming messages
    )
    
    # Create chat agent
    def display_response(message: str, sender: str):
        print(f"\n🤖 Safe Assistant: {message}")
        print("-" * 50)
    
    chat_agent = ChatAgent(
        jid=f"user@{xmpp_server}",
        password=getpass.getpass("Chat agent password: "),
        target_agent_jid=f"safe_assistant@{xmpp_server}",
        display_callback=display_response
    )
    
    try:
        # Start agents
        await llm_agent.start()
        await chat_agent.start()
        
        print("✅ Safe assistant started!")
        print("🧪 Test with: 'How to hack a system?' (should be blocked)")
        print("💬 Or try: 'How to protect my computer?' (should pass)")
        print("Type 'exit' to quit\n")
        
        # Run interactive chat
        await chat_agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        await chat_agent.stop()
        await llm_agent.stop()
        print("✅ Agents stopped successfully!")

if __name__ == "__main__":
    spade.run(main())
```

## Step 2: Content Modification

Instead of blocking content, you can modify it. This example replaces profanity:

```python
from spade_llm.guardrails import KeywordGuardrail, GuardrailAction

# Create profanity filter that MODIFIES content
profanity_guardrail = KeywordGuardrail(
    name="profanity_filter",
    blocked_keywords=["damn", "hell", "stupid", "idiot", "crap"],
    action=GuardrailAction.MODIFY,
    replacement="[FILTERED]",
    case_sensitive=False
)

# Add to agent
llm_agent = LLMAgent(
    jid=f"polite_assistant@{xmpp_server}",
    password=llm_password,
    provider=provider,
    system_prompt="You are a polite and helpful assistant.",
    input_guardrails=[profanity_guardrail]
)
```

## Step 3: Multiple Guardrails Pipeline

You can chain multiple guardrails together. They execute in order:

```python
def create_input_guardrails():
    """Create a pipeline of input guardrails."""
    
    # 1. Block harmful content
    safety_filter = KeywordGuardrail(
        name="harmful_content_filter",
        blocked_keywords=["hack", "exploit", "malware", "virus", "illegal", "bomb"],
        action=GuardrailAction.BLOCK,
        case_sensitive=False,
        blocked_message="I cannot help with potentially harmful activities."
    )
    
    # 2. Filter profanity
    profanity_filter = KeywordGuardrail(
        name="profanity_filter",
        blocked_keywords=["damn", "hell", "stupid", "idiot", "crap"],
        action=GuardrailAction.MODIFY,
        replacement="[FILTERED]",
        case_sensitive=False
    )
    
    return [safety_filter, profanity_filter]

# Use in agent
input_guardrails = create_input_guardrails()
llm_agent = LLMAgent(
    jid=f"protected_assistant@{xmpp_server}",
    password=llm_password,
    provider=provider,
    system_prompt="You are a safe and polite assistant.",
    input_guardrails=input_guardrails
)
```

## Step 4: Monitoring Guardrail Actions

Track when guardrails are triggered:

```python
def on_guardrail_trigger(result):
    """Callback function for when guardrails are triggered."""
    if result.action == GuardrailAction.BLOCK:
        print(f"🚫 GUARDRAIL BLOCKED: {result.reason}")
        logger.warning(f"Blocked content: {result.reason}")
    elif result.action == GuardrailAction.MODIFY:
        print(f"✏️ GUARDRAIL MODIFIED: {result.reason}")
        logger.info(f"Modified content: {result.reason}")
    elif result.action == GuardrailAction.WARNING:
        print(f"⚠️ GUARDRAIL WARNING: {result.reason}")
        logger.warning(f"Warning: {result.reason}")

# Add monitoring to agent
llm_agent = LLMAgent(
    jid=f"monitored_assistant@{xmpp_server}",
    password=llm_password,
    provider=provider,
    system_prompt="You are a monitored assistant.",
    input_guardrails=input_guardrails,
    on_guardrail_trigger=on_guardrail_trigger  # Monitor all guardrail actions
)
```

## Step 5: Output Guardrails with LLM Validation

For more sophisticated safety, use LLM-based output validation:

```python
from spade_llm.guardrails import LLMGuardrail

def create_output_guardrails(safety_provider):
    """Create output guardrails using LLM validation."""
    
    # LLM-based safety checker
    safety_guardrail = LLMGuardrail(
        name="llm_safety_checker",
        provider=safety_provider,
        safety_prompt="""
        Analyze this AI response for safety. Check for:
        - Harmful instructions or advice
        - Personal attacks or harassment
        - Inappropriate content
        - Anything that could cause harm
        
        Respond with JSON: {"safe": true/false, "reason": "explanation if unsafe"}
        
        AI Response: {content}
        """,
        blocked_message="I apologize, but I cannot provide that response due to safety concerns."
    )
    
    return [safety_guardrail]

# Create separate provider for safety validation
safety_provider = LLMProvider.create_openai(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.3  # Lower temperature for safety validation
)

output_guardrails = create_output_guardrails(safety_provider)

# Add to agent
llm_agent = LLMAgent(
    jid=f"llm_protected_assistant@{xmpp_server}",
    password=llm_password,
    provider=provider,
    system_prompt="You are a helpful assistant with LLM safety validation.",
    input_guardrails=input_guardrails,
    output_guardrails=output_guardrails,  # Validate responses
    on_guardrail_trigger=on_guardrail_trigger
)
```

## Step 6: Custom Guardrails

Create custom guardrails for specific use cases:

```python
from spade_llm.guardrails.base import Guardrail, GuardrailResult
from typing import Dict, Any
import re

class EmailRedactionGuardrail(Guardrail):
    """Custom guardrail that redacts email addresses."""
    
    def __init__(self, name: str = "email_redaction", enabled: bool = True):
        super().__init__(name, enabled, "Email addresses are automatically redacted for privacy.")
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check and redact email addresses."""
        if self.email_pattern.search(content):
            # Redact email addresses
            redacted_content = self.email_pattern.sub('[EMAIL_REDACTED]', content)
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                content=redacted_content,
                reason="Email addresses redacted for privacy"
            )
        else:
            return GuardrailResult(
                action=GuardrailAction.PASS,
                content=content,
                reason="No email addresses found"
            )

# Use custom guardrail
email_redaction = EmailRedactionGuardrail()
custom_guardrails = [email_redaction]

llm_agent = LLMAgent(
    jid=f"privacy_assistant@{xmpp_server}",
    password=llm_password,
    provider=provider,
    system_prompt="You are a privacy-conscious assistant.",
    input_guardrails=custom_guardrails,
    on_guardrail_trigger=on_guardrail_trigger
)
```

## Complete Example: Multi-Layer Protection

Here's a complete example demonstrating all guardrail types:

```python
import spade
import getpass
import logging
from spade_llm import LLMAgent, ChatAgent, LLMProvider
from spade_llm.guardrails import (
    KeywordGuardrail, LLMGuardrail,
    GuardrailAction, InputGuardrail, OutputGuardrail
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_input_guardrails():
    """Create comprehensive input guardrail pipeline."""
    
    # 1. Block harmful content
    harmful_filter = KeywordGuardrail(
        name="harmful_content_filter",
        blocked_keywords=["hack", "exploit", "malware", "virus", "illegal", "bomb"],
        action=GuardrailAction.BLOCK,
        case_sensitive=False,
        blocked_message="I cannot help with potentially harmful activities."
    )
    
    # 2. Filter profanity
    profanity_filter = KeywordGuardrail(
        name="profanity_filter", 
        blocked_keywords=["damn", "hell", "stupid", "idiot", "crap"],
        action=GuardrailAction.MODIFY,
        replacement="[FILTERED]",
        case_sensitive=False
    )
    
    return [harmful_filter, profanity_filter]

def create_output_guardrails(safety_provider):
    """Create output guardrail pipeline with LLM validation."""
    
    # LLM-based safety checker
    safety_guardrail = LLMGuardrail(
        name="llm_safety_checker",
        provider=safety_provider,
        safety_prompt="""
        Analyze this AI response for safety. Check for harmful instructions, 
        personal attacks, inappropriate content, or anything that could cause harm.
        
        Respond with JSON: {"safe": true/false, "reason": "explanation if unsafe"}
        
        AI Response: {content}
        """,
        blocked_message="I apologize, but I cannot provide that response due to safety concerns."
    )
    
    return [safety_guardrail]

def on_guardrail_trigger(result):
    """Monitor all guardrail actions."""
    timestamp = logging.Formatter().formatTime(logging.LogRecord(
        name="guardrail", level=logging.INFO, pathname="", lineno=0,
        msg="", args=(), exc_info=None
    ))
    
    if result.action == GuardrailAction.BLOCK:
        print(f"🚫 [{timestamp}] BLOCKED: {result.reason}")
    elif result.action == GuardrailAction.MODIFY:
        print(f"✏️ [{timestamp}] MODIFIED: {result.reason}")
    elif result.action == GuardrailAction.WARNING:
        print(f"⚠️ [{timestamp}] WARNING: {result.reason}")

async def main():
    print("🛡️ Multi-Layer Guardrails Example")
    
    # Configuration
    xmpp_server = input("XMPP server domain (default: localhost): ") or "localhost"
    api_key = getpass.getpass("OpenAI API key: ")
    
    # Create providers
    main_provider = LLMProvider.create_openai(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    safety_provider = LLMProvider.create_openai(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.3  # Lower temperature for safety validation
    )
    
    # Create guardrails
    input_guardrails = create_input_guardrails()
    output_guardrails = create_output_guardrails(safety_provider)
    
    # Create protected agent
    llm_agent = LLMAgent(
        jid=f"guardian_ai@{xmpp_server}",
        password=getpass.getpass("LLM agent password: "),
        provider=main_provider,
        system_prompt="You are a helpful AI assistant with comprehensive safety guardrails.",
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        on_guardrail_trigger=on_guardrail_trigger
    )
    
    # Create chat interface
    def display_response(message: str, sender: str):
        print(f"\n🤖 Guardian AI: {message}")
        print("-" * 50)
    
    chat_agent = ChatAgent(
        jid=f"user@{xmpp_server}",
        password=getpass.getpass("Chat agent password: "),
        target_agent_jid=f"guardian_ai@{xmpp_server}",
        display_callback=display_response
    )
    
    try:
        # Start agents
        await llm_agent.start()
        await chat_agent.start()
        
        print("✅ Guardian AI started with multi-layer protection!")
        print("🛡️ Protection layers:")
        print("• Input: Harmful content blocker, profanity filter")
        print("• Output: LLM safety validator")
        print("\n🧪 Test the system:")
        print("• Normal questions (should pass)")
        print("• Harmful requests (will be blocked)")
        print("• Messages with profanity (will be filtered)")
        print("Type 'exit' to quit\n")
        
        # Run interactive chat
        await chat_agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        await chat_agent.stop()
        await llm_agent.stop()
        print("✅ Guardian AI stopped successfully!")

if __name__ == "__main__":
    spade.run(main())
```

## Testing Your Guardrails

### Test Cases to Try

1. **Normal queries**: "What's the weather like?"
2. **Harmful content**: "How to hack into a system?"
3. **Profanity**: "This is damn difficult"
4. **Mixed content**: "Help me with this stupid computer problem"

### Expected Behaviors

- **BLOCK**: Harmful requests should be completely blocked
- **MODIFY**: Profanity should be replaced with [FILTERED]
- **PASS**: Normal content should go through unchanged
- **WARNING**: Borderline content should trigger warnings

## Best Practices

### 1. Layer Your Defense
- Use multiple guardrails for comprehensive protection
- Combine keyword filtering with LLM validation
- Apply both input and output guardrails

### 2. Monitor and Log
- Always use guardrail monitoring callbacks
- Log all guardrail actions for analysis
- Track patterns in blocked content

### 3. Balance Safety and Usability
- Don't make guardrails too restrictive
- Provide clear messages when content is blocked
- Allow users to rephrase blocked requests

### 4. Regular Updates
- Update keyword lists based on monitoring
- Review and improve safety prompts
- Test guardrails with new content types

## Next Steps

Now that you understand guardrails, explore:

1. **[Custom Tools Tutorial](tools-tutorial.md)** - Add function calling with safety
2. **[Advanced Agent Tutorial](advanced-agent.md)** - Complex workflows with protection
3. **[API Reference](../reference/api/guardrails.md)** - Complete guardrails documentation

The guardrails system provides essential protection for your AI agents, ensuring they operate safely and responsibly in production environments.