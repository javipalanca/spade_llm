# Human Interface API Reference

Complete API documentation for Human-in-the-Loop components.

## Overview

The Human Interface API consists of three main components:

- **`HumanInTheLoopTool`**: LLM tool for consulting human experts
- **`HumanInteractionBehaviour`**: SPADE behaviour for handling individual consultations  
- **Web Interface**: Browser-based interface for human experts

## HumanInTheLoopTool

LLM tool that enables agents to consult with human experts via XMPP messaging.

### Constructor

```python
HumanInTheLoopTool(
    human_expert_jid: str,
    timeout: float = 300.0,
    name: str = "ask_human_expert", 
    description: Optional[str] = None
)
```

**Parameters:**

- **`human_expert_jid`** (`str`): The XMPP JID of the human expert to consult
- **`timeout`** (`float`, optional): Maximum time to wait for response in seconds. Default: 300.0
- **`name`** (`str`, optional): Name of the tool for LLM reference. Default: "ask_human_expert"
- **`description`** (`str`, optional): Description of when to use this tool. Auto-generated if not provided.

**Example:**
```python
from spade_llm.tools import HumanInTheLoopTool

tool = HumanInTheLoopTool(
    human_expert_jid="expert@company.com",
    timeout=180.0,  # 3 minutes
    name="ask_domain_expert",
    description="Consult domain expert for specialized knowledge"
)
```

### Methods

#### `set_agent(agent: Agent)`

Binds the tool to an agent instance. Called automatically when tool is added to an agent.

**Parameters:**
- **`agent`** (`Agent`): The SPADE agent that will use this tool

**Note:** This method is called internally by `LLMAgent._register_tool()` and should not be called manually.

#### `_ask_human(question: str, context: Optional[str] = None) -> str` 

Internal method that executes the human consultation. Called by the LLM system.

**Parameters:**
- **`question`** (`str`): The question to ask the human expert
- **`context`** (`str`, optional): Additional context to help the human understand

**Returns:**
- **`str`**: The human expert's response or an error message

**Raises:**
- **`TimeoutError`**: If human doesn't respond within the timeout period
- **`Exception`**: For XMPP connection or other system errors

## HumanInteractionBehaviour

SPADE OneShotBehaviour that handles individual human consultations via XMPP.

### Constructor

```python
HumanInteractionBehaviour(
    human_jid: str,
    question: str,
    context: Optional[str] = None,
    timeout: float = 300.0
)
```

**Parameters:**

- **`human_jid`** (`str`): The XMPP JID of the human expert
- **`question`** (`str`): The question to ask the human
- **`context`** (`str`, optional): Additional context for the question
- **`timeout`** (`float`, optional): Maximum wait time for response. Default: 300.0

**Example:**
```python
from spade_llm.behaviour.human_interaction import HumanInteractionBehaviour

behaviour = HumanInteractionBehaviour(
    human_jid="expert@company.com",
    question="What's our policy on remote work?",
    context="New employee asking about work arrangement options",
    timeout=240.0
)
```

### Attributes

#### `query_id: str`

Unique 8-character identifier for this consultation, used for message correlation.

#### `response: Optional[str]`

The human expert's response. Set after successful completion, `None` if no response received.

### Methods

#### `async run()`

Executes the behaviour: sends question to human and waits for response.

**Workflow:**
1. Waits for agent XMPP connection
2. Formats and sends question with unique thread ID
3. Waits for response with timeout
4. Stores response in `self.response`

**Note:** This method is called automatically by SPADE and should not be invoked manually.

#### `_format_question() -> str`

Formats the question for display to the human expert.

**Returns:**
- **`str`**: Formatted question with query ID, context, and response instructions

**Example output:**
```
[Query a1b2c3d4] What's our policy on remote work?

Context: New employee asking about work arrangement options

(Please reply to this message to provide your answer)
```

## Web Interface

The web interface consists of static files served by a simple HTTP server.

### Web Server

#### `run_server(port=8080, directory=None)`

Starts the HTTP server for the human expert interface.

**Parameters:**
- **`port`** (`int`, optional): Port to run server on. Default: 8080
- **`directory`** (`str`, optional): Directory to serve files from. Default: `web_client` folder

**Example:**
```python
from spade_llm.human_interface.web_server import run_server

# Start on default port
run_server()

# Start on custom port
run_server(port=9000)
```

### Starting via Command Line

```bash
# Default configuration (port 8080)
python -m spade_llm.human_interface.web_server

# Custom port
python -m spade_llm.human_interface.web_server 9000
```

### Web Interface Features

#### Connection Management

- **WebSocket connection** to XMPP server
- **Automatic reconnection** on connection loss
- **Visual connection status** indicator

#### Query Handling

- **Real-time notifications** for new queries
- **Query filtering** (show/hide answered)
- **Response history** tracking
- **Thread-based correlation** for proper message routing

#### User Interface Elements

- **Connection form**: XMPP credentials input
- **Query list**: Active and historical queries  
- **Response interface**: Text area and send button
- **Status indicators**: Connection and query states

## Integration Patterns

### Agent Registration

```python
from spade_llm.agent import LLMAgent
from spade_llm.tools import HumanInTheLoopTool

# Create tool
human_tool = HumanInTheLoopTool("expert@company.com")

# Method 1: Pass in constructor
agent = LLMAgent(
    jid="agent@company.com",
    password="password",
    provider=provider,
    tools=[human_tool]  # Automatic registration and binding
)

# Method 2: Add after creation
agent = LLMAgent(jid="agent@company.com", password="password", provider=provider)
agent.add_tool(human_tool)  # Manual registration
```

### Error Handling Patterns

```python
# In tool usage
try:
    response = await human_tool._ask_human("What's the weather?")
    if response.startswith("Timeout:"):
        # Handle timeout gracefully
        return "Human expert unavailable, using alternative approach"
    elif response.startswith("Error:"):
        # Handle system errors
        return "Consultation failed, proceeding with available information"
    else:
        return f"Expert says: {response}"
except Exception as e:
    return f"Unexpected error: {e}"
```

### Multiple Expert Configuration

```python
# Domain-specific experts
sales_expert = HumanInTheLoopTool(
    human_expert_jid="sales@company.com",
    name="ask_sales_expert",
    description="Consult sales team about pricing and customers"
)

tech_expert = HumanInTheLoopTool(
    human_expert_jid="tech@company.com", 
    name="ask_tech_expert",
    description="Consult tech team about systems and architecture"
)

agent = LLMAgent(
    jid="agent@company.com",
    password="password", 
    provider=provider,
    tools=[sales_expert, tech_expert],
    system_prompt="""Choose the appropriate expert:
    - Use ask_sales_expert for pricing, deals, customer questions
    - Use ask_tech_expert for technical, system, architecture questions"""
)
```

## Message Protocol

### XMPP Message Format

#### Question Message (Agent → Human)

```xml
<message to="expert@company.com" type="chat" thread="a1b2c3d4">
  <body>
    [Query a1b2c3d4] What's our WiFi password?
    
    Context: New employee needs network access
    
    (Please reply to this message to provide your answer)
  </body>
  <metadata type="human_query" query_id="a1b2c3d4"/>
</message>
```

#### Response Message (Human → Agent)

```xml
<message to="agent@company.com" type="chat" thread="a1b2c3d4">
  <body>The WiFi password is "CompanyWiFi2024" - please don't share externally.</body>
</message>
```

### Message Correlation

Messages are correlated using XMPP thread IDs:

1. **Query ID generated**: 8-character UUID segment
2. **Thread set**: `msg.thread = query_id` 
3. **Response inherits**: Automatic thread inheritance in XMPP
4. **Behaviour filters**: Only processes messages with matching thread

## Configuration Reference

### Environment Variables

```bash
# XMPP server configuration
XMPP_SERVER=your-server.com
XMPP_WEBSOCKET_PORT=7070

# Expert credentials
EXPERT_JID=expert@your-server.com  
EXPERT_PASSWORD=secure-password

# Web interface
WEB_INTERFACE_PORT=8080
```

### System Prompt Examples

#### Basic Configuration

```python
system_prompt = """You are an AI assistant with access to human experts.

Use the ask_human_expert tool when you need:
- Current information not in your training data
- Human judgment or opinions
- Company-specific information  
- Clarification on ambiguous requests

Always explain whether information comes from human experts or your training."""
```

#### Advanced Configuration

```python
system_prompt = """You are an enterprise AI assistant with specialized expert access.

Expert Consultation Rules:
1. ask_sales_expert: pricing, deals, customer relationships, market intelligence
2. ask_tech_expert: architecture, systems, security, technical feasibility  
3. ask_legal_expert: compliance, contracts, regulatory questions
4. ask_hr_expert: policies, procedures, employee-related questions

Only consult experts for:
- Information requiring current knowledge (post-training cutoff)
- Subjective decisions requiring human judgment
- Company-specific policies or procedures
- Sensitive matters requiring approval

For general knowledge questions, answer directly without consultation."""
```

## Security Considerations

### Access Control

- **XMPP authentication**: All participants must authenticate to XMPP server
- **JID validation**: Tools validate expert JID format  
- **Message encryption**: Use XMPP TLS/SSL in production
- **Web interface**: No authentication by default - add auth layer for production

### Data Privacy

- **Message logging**: XMPP servers may log messages - configure retention policies
- **Browser storage**: Web interface uses sessionStorage (cleared on close)
- **Network traffic**: Use encrypted WebSocket connections (`wss://`) in production

### Deployment Security

```python
# Production configuration example
human_tool = HumanInTheLoopTool(
    human_expert_jid=os.getenv("EXPERT_JID"),  # From environment
    timeout=int(os.getenv("EXPERT_TIMEOUT", "300")),
    name="ask_expert",
    description="Consult verified human expert for sensitive information"
)

# Agent with security enabled
agent = LLMAgent(
    jid=os.getenv("AGENT_JID"),
    password=os.getenv("AGENT_PASSWORD"),
    provider=provider,
    tools=[human_tool],
    verify_security=True  # Enable certificate verification
)
```

## Performance Considerations

### Timeout Strategy

```python
# Tiered timeout approach
urgent_tool = HumanInTheLoopTool("expert@company.com", timeout=60.0)   # 1 min
normal_tool = HumanInTheLoopTool("expert@company.com", timeout=300.0)  # 5 min  
research_tool = HumanInTheLoopTool("expert@company.com", timeout=1800.0) # 30 min
```

### Connection Pooling

```python
# Reuse connections for multiple consultations
agent = LLMAgent(...)  # Single agent instance
await agent.start()     # Establish XMPP connection once

# Multiple consultations reuse the same agent connection
for question in questions:
    response = await agent.process_message(question)
```

### Expert Availability

```python
# Fallback strategy
system_prompt = """When consulting experts:
1. If timeout occurs, inform user and provide best-effort answer
2. If expert unavailable, use available information sources
3. Set user expectations about response times
4. For urgent matters, escalate through alternative channels"""
```