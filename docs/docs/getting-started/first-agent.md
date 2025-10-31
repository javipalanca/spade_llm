# Your First LLM Agent

In this tutorial, you'll learn how to create your first SPADE-LLM agent step by step. We'll start with a basic setup and gradually add features to understand the core concepts.

## Prerequisites

Before starting, ensure you have:
- Python 3.10 or higher installed
- SPADE-LLM installed (`pip install spade_llm`)
- **SPADE's built-in server running** (recommended - no external setup needed!)
- Access to at least one LLM provider (OpenAI API key or local Ollama installation)

### Start SPADE Server

**New in SPADE 4.0+ - Built-in XMPP server included!**

```bash
# Terminal 1: Start SPADE's built-in server
spade run
```

This eliminates the need for external XMPP servers like Prosody. Keep this running in a separate terminal while you work through the tutorial.

## Step 1: Basic Agent Setup

Let's start with the simplest possible SPADE-LLM agent. Create `my_first_agent.py`:
This agent does not have the capability to interact with us except through XMPP messages. If we want to chat with it we need to use 
ChatAgent (next step)
```python
import spade
from spade_llm import LLMAgent, LLMProvider

async def main():
    # Create an LLM provider
    provider = LLMProvider.create_openai(
        api_key="your-api-key-here",
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create the LLM agent (using SPADE's built-in server)
    agent = LLMAgent(
        jid="assistant@localhost",
        password="password123",
        provider=provider,
        system_prompt="You are a helpful assistant.",
    )
    
    # Start the agent
    await agent.start()
    print("✅ Agent started successfully!")
    

if __name__ == "__main__":
    spade.run(main())
```

### Understanding the Components

**LLMProvider**: Your interface to different LLM services. SPADE-LLM supports multiple providers:

```python
# OpenAI
provider = LLMProvider.create_openai(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Ollama (local)
provider = LLMProvider.create_ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1"
)

# LM Studio (local)
provider = LLMProvider.create_lm_studio(
    model="local-model",
    base_url="http://localhost:1234/v1"
)
```

**LLMAgent**: The core component that connects to XMPP, receives messages, processes them through the LLM, and sends responses back.

Run with: `python my_first_agent.py`

## Step 2: Creating an Interactive Chat

To make your agent interactive, you'll need a `ChatAgent` to handle user input. Create `interactive_agent.py`:

```python
import spade
import getpass
from spade_llm import LLMAgent, ChatAgent, LLMProvider

async def main():
    # Using SPADE's built-in server (make sure it's running!)
    spade_server = "localhost"
    
    print("🚀 Using SPADE's built-in server")
    print("Make sure you started it with: spade run")
    input("Press Enter when the server is running...")
    
    # Create LLM provider
    provider = LLMProvider.create_openai(
        api_key="your-api-key",
        model="gpt-4o-mini"
    )
    
    # Create the LLM agent
    llm_agent = LLMAgent(
        jid=f"assistant@{spade_server}",
        password="assistant_pass",  # Simple password for built-in server
        provider=provider,
        system_prompt="You are a helpful assistant. Keep responses concise and friendly.",
    )
    
    # Create the chat agent for user interaction
    chat_agent = ChatAgent(
        jid=f"user@{spade_server}",
        password="user_pass",  # Simple password for built-in server
        target_agent_jid=f"assistant@{spade_server}",
    )
    
    try:
        # Start both agents
        await llm_agent.start()
        await chat_agent.start()
        
        print("✅ Agents started successfully!")
        print("💬 You can now chat with your AI assistant")
        print("Type 'exit' to quit\n")
        
        # Run interactive chat
        await chat_agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        # Clean up
        await chat_agent.stop()
        await llm_agent.stop()
        print("✅ Agents stopped successfully!")

if __name__ == "__main__":
    spade.run(main())
```

### Adding Custom Display

You can customize how responses are displayed:

```python
def display_response(message: str, sender: str):
    print(f"\n🤖 Assistant: {message}")
    print("-" * 50)

def on_message_sent(message: str, recipient: str):
    print(f"👤 You: {message}")

chat_agent = ChatAgent(
    jid=f"user@localhost",
    password="user_pass",
    target_agent_jid=f"assistant@localhost",
    display_callback=display_response,
    on_message_sent=on_message_sent
)
```

## Step 3: Adding Error Handling

Always include proper error handling in production code:

```python
import spade
import logging
from spade_llm import LLMAgent, ChatAgent, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Create provider with error handling
        provider = LLMProvider.create_openai(
            api_key="your-api-key",
            model="gpt-4o-mini",
            timeout=30.0  # Set timeout
        )
        
        # Create agents with error handling
        llm_agent = LLMAgent(
            jid="assistant@localhost",
            password="password123",
            provider=provider,
            system_prompt="You are a helpful assistant."
        )
        
        await llm_agent.start()
        logger.info("✅ LLM Agent started successfully")
        
        # Your chat logic here...
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print("💡 Check your configuration and try again")
    finally:
        # Always clean up
        if 'llm_agent' in locals():
            await llm_agent.stop()
        if 'chat_agent' in locals():
            await chat_agent.stop()

if __name__ == "__main__":
    spade.run(main())
```

## Complete Example with Best Practices

Here's a complete, production-ready example that demonstrates best practices:

```python
import spade
import getpass
import logging
from spade_llm import LLMAgent, ChatAgent, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("🚀 Starting your first SPADE-LLM agent!")
    print("📋 Make sure SPADE server is running: spade run")
    input("Press Enter when the server is running...")
    
    # Configuration - using built-in SPADE server
    spade_server = "localhost"
    
    # Create provider (choose one)
    provider_type = input("Provider (openai/ollama): ").lower()
    
    if provider_type == "openai":
        api_key = getpass.getpass("OpenAI API key: ")
        provider = LLMProvider.create_openai(
            api_key=api_key,
            model="gpt-4o-mini",
            timeout=30.0
        )
    else:  # ollama
        model = input("Ollama model (default: llama3.1:8b): ") or "llama3.1:8b"
        provider = LLMProvider.create_ollama(
            model=model,
            base_url="http://localhost:11434/v1",
            timeout=60.0
        )
    
    # Simple passwords for built-in server (no need for getpass)
    llm_password = "assistant_pass"
    chat_password = "user_pass"
    
    # Create agents
    llm_agent = LLMAgent(
        jid=f"assistant@{spade_server}",
        password=llm_password,
        provider=provider,
        system_prompt="You are a helpful and friendly AI assistant. Keep responses concise but informative."
    )
    
    def display_response(message: str, sender: str):
        print(f"\n🤖 Assistant: {message}")
        print("-" * 50)
    
    def on_message_sent(message: str, recipient: str):
        print(f"👤 You: {message}")
    
    chat_agent = ChatAgent(
        jid=f"user@{spade_server}",
        password=chat_password,
        target_agent_jid=f"assistant@{spade_server}",
        display_callback=display_response,
        on_message_sent=on_message_sent
    )
    
    try:
        # Start agents
        await llm_agent.start()
        await chat_agent.start()
        
        logger.info("✅ Agents started successfully!")
        print("💬 Start chatting with your AI assistant")
        print("Type 'exit' to quit\n")
        
        # Run interactive chat
        await chat_agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print("💡 Check your configuration and try again")
    finally:
        # Clean up
        try:
            await chat_agent.stop()
            await llm_agent.stop()
            logger.info("✅ Agents stopped successfully!")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    spade.run(main())
```

## Testing Your Agent

1. **Start SPADE server**: `spade run`
2. **Run the agent**: `python my_first_agent.py`
3. **Choose your provider**: OpenAI or Ollama
4. **Start chatting**: Type messages and get responses

## Common Issues and Solutions

### SPADE Server Issues
- **Server won't start**: Check if port 5222 is already in use (`netstat -an | grep 5222`)
- **Port conflicts**: Try different ports: `spade run --client_port 6222 --server_port 6269`
- **Agent connection fails**: Ensure server is running before starting agents

### Connection Problems
- **Agent won't start**: Ensure SPADE built-in server is running first
- **Authentication failed**: Built-in server auto-registers agents, but verify JID format
- **Network issues**: Built-in server runs locally, check firewall settings

### LLM Provider Issues
- **OpenAI errors**: Verify API key and account credits
- **Ollama not responding**: Check if `ollama serve` is running
- **Timeout errors**: Increase timeout values for slow models

### Performance Tips
- Use appropriate model sizes for your hardware
- Set reasonable timeouts based on your provider
- Monitor token usage for cost optimization
- Use local models (Ollama) for development

## Next Steps

Now that you have a working agent, explore these advanced features:

1. **[Custom Tools Tutorial](tools-tutorial.md)** - Add function calling capabilities
2. **[Guardrails Tutorial](guardrails-tutorial.md)** - Implement safety and content filtering  
3. **[Advanced Agent Tutorial](advanced-agent.md)** - Multi-agent workflows and integrations

Each tutorial builds on the concepts you've learned here, gradually adding more sophisticated capabilities to your agents.
