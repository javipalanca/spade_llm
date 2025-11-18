# Custom Tools Tutorial

In this tutorial, you'll learn how to create and use custom tools with your SPADE-LLM agents. Tools enable your agents to perform actions beyond text generation, such as retrieving data, performing calculations, or interacting with external services.

## What are Tools?

Tools are Python functions that your LLM agent can call to perform specific tasks. When the LLM needs to execute an action, it can:

1. **Identify** which tool to use
2. **Extract** the required parameters
3. **Execute** the tool function
4. **Use** the results to generate a response

## Prerequisites

- Complete the [First Agent Tutorial](first-agent.md)
- Basic understanding of Python functions
- SPADE-LLM installed with all dependencies
- Access to an LLM provider that supports function calling

## Step 1: Basic Tool Creation

Let's start with simple tools. Based on the `ollama_with_tools_example.py`, here's how to create basic tools:

```python
import spade
import getpass
from datetime import datetime
from spade_llm import LLMAgent, ChatAgent, LLMProvider, LLMTool

# Simple tool functions
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_math(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow basic math operations for safety
        allowed_names = {
            k: v for k, v in __builtins__.items() 
            if k in ['abs', 'round', 'min', 'max', 'sum']
        }
        result = eval(expression, {"__builtins__": allowed_names})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_weather(city: str) -> str:
    """Get simulated weather for a city."""
    weather_data = {
        "madrid": "22°C, sunny",
        "london": "15°C, cloudy",
        "new york": "18°C, rainy",
        "tokyo": "25°C, clear",
        "paris": "19°C, partly cloudy",
        "barcelona": "24°C, sunny"
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")

async def main():
    print("🔧 Custom Tools Tutorial: Basic Tool Creation")
    
    # Configuration
    xmpp_server = input("XMPP server domain (default: localhost): ") or "localhost"
    
    # Create provider (using Ollama as in the example)
    provider = LLMProvider.create_ollama(
        model="qwen2.5:7b",  # Or any model that supports function calling
        base_url="http://localhost:11434/v1",
        temperature=0.7
    )
    
    # Create tools with proper schema definitions
    tools = [
        LLMTool(
            name="get_current_time",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=get_current_time
        ),
        LLMTool(
            name="calculate_math",
            description="Safely evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                    }
                },
                "required": ["expression"]
            },
            func=calculate_math
        ),
        LLMTool(
            name="get_weather",
            description="Get weather information for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'Madrid', 'London')"
                    }
                },
                "required": ["city"]
            },
            func=get_weather
        )
    ]
    
    # Create LLM agent with tools
    llm_agent = LLMAgent(
        jid=f"tool_assistant@{xmpp_server}",
        password=getpass.getpass("LLM agent password: "),
        provider=provider,
        system_prompt="You are a helpful assistant with access to tools: get_current_time, calculate_math, and get_weather. Use these tools when appropriate to help users.",
        tools=tools  # Pass tools to the agent
    )
    
    # Create chat interface
    def display_response(message: str, sender: str):
        print(f"\n🤖 Tool Assistant: {message}")
        print("-" * 50)
    
    chat_agent = ChatAgent(
        jid=f"user@{xmpp_server}",
        password=getpass.getpass("Chat agent password: "),
        target_agent_jid=f"tool_assistant@{xmpp_server}",
        display_callback=display_response
    )
    
    try:
        # Start agents
        await llm_agent.start()
        await chat_agent.start()
        
        print("✅ Tool assistant started!")
        print("🔧 Available tools:")
        print("• get_current_time - Get current date and time")
        print("• calculate_math - Perform mathematical calculations")
        print("• get_weather - Get weather for major cities")
        print("\n💡 Try these queries:")
        print("• 'What time is it?'")
        print("• 'Calculate 15 * 8 + 32'")
        print("• 'What's the weather in Madrid?'")
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
    print("🚀 Prerequisites:")
    print("• Ollama running: ollama serve")
    print("• Model available: ollama pull qwen2.5:7b")
    print("• XMPP server running")
    print()
    
    spade.run(main())
```

## Step 2: Understanding Tool Schemas

Tool schemas define what parameters your tools accept. Here's the structure:

```python
# Tool schema follows JSON Schema format
tool_schema = {
    "type": "object",
    "properties": {
        "parameter_name": {
            "type": "string",  # or "number", "boolean", "array", "object"
            "description": "Clear description of what this parameter does"
        }
    },
    "required": ["parameter_name"]  # List of required parameters
}
```

### Examples of Different Parameter Types:

```python
# Simple string parameter
simple_tool = LLMTool(
    name="greet_user",
    description="Greet a user by name",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User's name"}
        },
        "required": ["name"]
    },
    func=lambda name: f"Hello, {name}!"
)

# Multiple parameters with different types
complex_tool = LLMTool(
    name="create_reminder",
    description="Create a reminder with specific details",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "minutes": {"type": "number", "description": "Minutes from now"},
            "priority": {
                "type": "string", 
                "enum": ["low", "medium", "high"],
                "description": "Priority level"
            },
            "urgent": {"type": "boolean", "description": "Is this urgent?"}
        },
        "required": ["title", "minutes"]
    },
    func=create_reminder_func
)
```


## Complete Example: Multi-Tool Agent

Here's a complete example combining all tool types:

```python
import spade
import getpass
import logging
from datetime import datetime
from spade_llm import LLMAgent, ChatAgent, LLMProvider, LLMTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool memory for data persistence
tool_memory = {}

# Basic utility tools
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_math(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Safe evaluation with limited built-ins
        safe_dict = {"__builtins__": {}}
        safe_dict.update({name: getattr(__builtins__, name, None) 
                         for name in ['abs', 'round', 'min', 'max', 'sum', 'len']})
        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"Math error: {str(e)}"

def get_weather(city: str) -> str:
    """Get weather information for major cities."""
    weather_data = {
        "madrid": "22°C, sunny with light clouds",
        "london": "15°C, cloudy with occasional rain",
        "new york": "18°C, rainy with strong winds",
        "tokyo": "25°C, clear skies",
        "paris": "19°C, partly cloudy",
        "barcelona": "24°C, sunny and warm",
        "berlin": "16°C, overcast",
        "rome": "26°C, sunny"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

# Memory tools for data persistence
def store_note(title: str, content: str) -> str:
    """Store a note in memory."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tool_memory[title] = {
        "content": content,
        "timestamp": timestamp
    }
    return f"Note '{title}' stored successfully"

def retrieve_note(title: str) -> str:
    """Retrieve a previously stored note."""
    if title in tool_memory:
        note = tool_memory[title]
        return f"Note '{title}' (stored: {note['timestamp']}):\n{note['content']}"
    else:
        return f"No note found with title '{title}'"

def list_notes() -> str:
    """List all stored notes."""
    if tool_memory:
        notes = [f"'{title}' (stored: {note['timestamp']})" 
                for title, note in tool_memory.items()]
        return f"Stored notes:\n" + "\n".join(notes)
    else:
        return "No notes stored"

# Create comprehensive tool set
def create_tools():
    """Create a comprehensive set of tools."""
    return [
        # Utility tools
        LLMTool(
            name="get_current_time",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=get_current_time
        ),
        LLMTool(
            name="calculate_math",
            description="Perform mathematical calculations safely",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression (e.g., '2 + 3 * 4', 'round(15.7)')"
                    }
                },
                "required": ["expression"]
            },
            func=calculate_math
        ),
        LLMTool(
            name="get_weather",
            description="Get weather information for major cities",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (Madrid, London, New York, Tokyo, Paris, Barcelona, Berlin, Rome)"
                    }
                },
                "required": ["city"]
            },
            func=get_weather
        ),
        
        # Memory tools
        LLMTool(
            name="store_note",
            description="Store a note in memory for later retrieval",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the note"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the note"
                    }
                },
                "required": ["title", "content"]
            },
            func=store_note
        ),
        LLMTool(
            name="retrieve_note",
            description="Retrieve a previously stored note",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the note to retrieve"
                    }
                },
                "required": ["title"]
            },
            func=retrieve_note
        ),
        LLMTool(
            name="list_notes",
            description="List all stored notes",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=list_notes
        )
    ]

async def main():
    print("🔧 Complete Multi-Tool Agent Example")
    
    # Configuration
    xmpp_server = input("XMPP server domain (default: localhost): ") or "localhost"
    
    # Choose provider
    provider_type = input("Provider (openai/ollama): ").lower()
    
    if provider_type == "openai":
        provider = LLMProvider.create_openai(
            api_key=getpass.getpass("OpenAI API key: "),
            model="gpt-4o-mini",
            temperature=0.7
        )
    else:  # ollama
        model = input("Ollama model (default: qwen2.5:7b): ") or "qwen2.5:7b"
        provider = LLMProvider.create_ollama(
            model=model,
            base_url="http://localhost:11434/v1",
            temperature=0.7
        )
    
    # Create tools
    tools = create_tools()
    
    # Create multi-tool agent
    llm_agent = LLMAgent(
        jid=f"multi_tool_agent@{xmpp_server}",
        password=getpass.getpass("LLM agent password: "),
        provider=provider,
        system_prompt="""You are a helpful assistant with access to multiple tools:

UTILITY TOOLS:
- get_current_time: Get current date and time
- calculate_math: Perform mathematical calculations  
- get_weather: Get weather for major cities

MEMORY TOOLS:
- store_note: Store information for later retrieval
- retrieve_note: Get previously stored notes
- list_notes: See all stored notes

Use these tools appropriately to help users. When using tools, explain what you're doing and why.""",
        tools=tools
    )
    
    # Create chat interface
    def display_response(message: str, sender: str):
        print(f"\n🤖 Multi-Tool Agent: {message}")
        print("-" * 60)
    
    chat_agent = ChatAgent(
        jid=f"user@{xmpp_server}",
        password=getpass.getpass("Chat agent password: "),
        target_agent_jid=f"multi_tool_agent@{xmpp_server}",
        display_callback=display_response
    )
    
    try:
        # Start agents
        await llm_agent.start()
        await chat_agent.start()
        
        print("✅ Multi-Tool Agent started!")
        print("🔧 Available tools:")
        print("  Utility: time, math, weather")
        print("  Memory: store/retrieve/list notes")
        print("\n💡 Example queries:")
        print("• 'What time is it and what's the weather in Madrid?'")
        print("• 'Calculate 15 * 8 + 32 and store the result as a note'")
        print("• 'Store a note about my meeting tomorrow'")
        print("• 'Show me all my notes'")
        print("Type 'exit' to quit\n")
        
        # Run interactive chat
        await chat_agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        await chat_agent.stop()
        await llm_agent.stop()
        print("✅ Multi-Tool Agent stopped!")

if __name__ == "__main__":
    spade.run(main())
```

## Best Practices

### 1. Tool Design
- Keep tools focused on single responsibilities
- Use clear, descriptive names and documentation
- Implement comprehensive error handling
- Validate inputs and sanitize outputs

### 2. Schema Definition
- Provide detailed parameter descriptions
- Use appropriate data types
- Mark required parameters clearly
- Include examples in descriptions

## Next Steps

Now that you understand custom tools, explore:

1. **[Advanced Agent Tutorial](advanced-agent.md)** - Multi-agent workflows with tools
2. **[MCP Integration Guide](../guides/mcp.md)** - External service integration
3. **[API Reference](../reference/api/tools.md)** - Complete tools documentation

Custom tools are powerful building blocks that extend your agents' capabilities beyond text generation, enabling them to interact with the world and perform real tasks.