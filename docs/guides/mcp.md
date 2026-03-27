# MCP Integration

Model Context Protocol (MCP) enables SPADE_LLM agents to connect to external services and tools through standardized servers.

## Overview

MCP provides a standard way for AI applications to connect to data sources and tools. SPADE_LLM automatically discovers and adapts MCP tools for use with LLM agents.

### Benefits

- **Standardized Interface**: Consistent API across different services
- **Dynamic Discovery**: Automatic tool detection from MCP servers
- **External Services**: Connect to databases, APIs, and file systems
- **Tool Caching**: Improved performance through tool caching

## MCP Server Types

### STDIO Servers

Communicate via standard input/output streams:

```python
from spade_llm.mcp import StdioServerConfig

server_config = StdioServerConfig(
    name="DatabaseServer",
    command="python",
    args=["path/to/database_server.py"],
    env={"DB_URL": "sqlite:///data.db"},
    cache_tools=True
)
```

### SSE Servers

!!! warning "SSE Transport Deprecated"
    The SSE transport is deprecated in favor of the new Streamable HTTP transport. 
    While SSE is still supported for backward compatibility, new implementations 
    should use `StreamableHttpServerConfig` instead of `SseServerConfig`.

Communicate via Server-Sent Events over HTTP:

```python
from spade_llm.mcp import SseServerConfig

server_config = SseServerConfig(
    name="WebService",
    url="http://localhost:8080/mcp",
    cache_tools=True
)
```

### Streamable HTTP Servers

Communicate via the modern Streamable HTTP protocol (recommended for new implementations):

```python
from spade_llm.mcp import StreamableHttpServerConfig

server_config = StreamableHttpServerConfig(
    name="ModernWebService",
    url="http://localhost:8080/mcp",
    headers={"Authorization": "Bearer token"},
    timeout=30.0,  # Connection timeout in seconds
    sse_read_timeout=300.0,  # Read timeout for SSE stream in seconds
    terminate_on_close=True,  # Terminate connection on close
    cache_tools=True
)
```

The Streamable HTTP transport provides:
- Improved session management and stability
- Better handling of long-running connections
- Enhanced error recovery mechanisms
- Full compatibility with the MCP specification

## Basic Usage

### Agent with MCP Tools

```python
import spade
from spade_llm import LLMAgent, LLMProvider
from spade_llm.mcp import StdioServerConfig

async def main():
    # Configure MCP server
    mcp_server = StdioServerConfig(
        name="FileManager",
        command="python",
        args=["-m", "file_manager_mcp"],
        cache_tools=True
    )
    
    # Create agent with MCP integration
    agent = LLMAgent(
        jid="assistant@example.com",
        password="password",
        provider=provider,
        system_prompt="You are a helpful assistant with file management capabilities",
        mcp_servers=[mcp_server]
    )
    
    await agent.start()

if __name__ == "__main__":
    spade.run(main())
```

### Multiple MCP Servers

```python
# Configure multiple servers
mcp_servers = [
    StdioServerConfig(
        name="DatabaseService",
        command="python",
        args=["database_mcp_server.py"],
        env={"DB_CONNECTION": "postgresql://localhost/mydb"}
    ),
    StdioServerConfig(
        name="WeatherService", 
        command="node",
        args=["weather_mcp_server.js"],
        env={"API_KEY": "your-weather-api-key"}
    ),
    StreamableHttpServerConfig(
        name="AnalyticsService",
        url="https://analytics.example.com/mcp",
        headers={"X-API-Key": "your-api-key"},
        terminate_on_close=True,
        cache_tools=True
    )
]

agent = LLMAgent(
    jid="multi-service@example.com",
    password="password",
    provider=provider,
    mcp_servers=mcp_servers
)
```

## Creating MCP Servers

### Basic STDIO Server

Create a simple MCP server (`math_server.py`):

```python
#!/usr/bin/env python3
import json
import sys
import math
from typing import Any, Dict

class MathMCPServer:
    def __init__(self):
        self.tools = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "sqrt",
                "description": "Calculate square root",
                "inputSchema": {
                    "type": "object", 
                    "properties": {
                        "number": {
                            "type": "number",
                            "description": "Number to calculate square root of"
                        }
                    },
                    "required": ["number"]
                }
            }
        ]
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests."""
        method = request.get("method")
        
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"tools": self.tools}
            }
        
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "calculate":
                return self._calculate(request.get("id"), arguments)
            elif tool_name == "sqrt":
                return self._sqrt(request.get("id"), arguments)
            
            return self._error(request.get("id"), "Unknown tool")
        
        elif method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "math-server",
                        "version": "1.0.0"
                    }
                }
            }
        
        return self._error(request.get("id"), "Unknown method")
    
    def _calculate(self, request_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate tool call."""
        try:
            expression = args.get("expression", "")
            # Safe evaluation (restrict to math operations)
            result = eval(expression, {"__builtins__": {}, "math": math})
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text", 
                            "text": f"Result: {result}"
                        }
                    ]
                }
            }
        except Exception as e:
            return self._error(request_id, f"Calculation error: {str(e)}")
    
    def _sqrt(self, request_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sqrt tool call."""
        try:
            number = float(args.get("number", 0))
            if number < 0:
                return self._error(request_id, "Cannot calculate square root of negative number")
            
            result = math.sqrt(number)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"√{number} = {result}"
                        }
                    ]
                }
            }
        except Exception as e:
            return self._error(request_id, f"Square root error: {str(e)}")
    
    def _error(self, request_id: str, message: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -1,
                "message": message
            }
        }
    
    def run(self):
        """Run the MCP server."""
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }
                print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    server = MathMCPServer()
    server.run()
```

### Using the Math Server

```python
# Configure the math MCP server
math_server = StdioServerConfig(
    name="MathService",
    command="python",
    args=["math_server.py"],
    cache_tools=True
)

# Create agent with math capabilities
agent = LLMAgent(
    jid="math-assistant@example.com",
    password="password",
    provider=provider,
    system_prompt="You are a math assistant. Use the calculate and sqrt tools for mathematical operations.",
    mcp_servers=[math_server]
)

await agent.start()
```

## Configuration Options

### Server Configuration

```python
# STDIO Server with environment variables
stdio_config = StdioServerConfig(
    name="MyService",
    command="python",
    args=["my_mcp_server.py"],
    env={
        "API_KEY": "your-api-key",
        "DB_URL": "postgresql://localhost/mydb",
        "LOG_LEVEL": "INFO"
    },
    cache_tools=True,
    working_directory="/path/to/server"
)

# SSE Server with authentication (deprecated - use Streamable HTTP instead)
sse_config = SseServerConfig(
    name="WebService",
    url="https://api.example.com/mcp",
    headers={
        "Authorization": "Bearer your-token",
        "X-API-Version": "v1"
    },
    cache_tools=True
)

# Streamable HTTP Server with advanced options (recommended)
streamable_config = StreamableHttpServerConfig(
    name="ModernService",
    url="https://api.example.com/mcp",
    headers={
        "Authorization": "Bearer your-token",
        "X-API-Version": "v2"
    },
    timeout=30.0,  # Connection timeout in seconds
    sse_read_timeout=300.0,  # Read timeout for SSE stream (5 minutes)
    terminate_on_close=True,  # Cleanly terminate connection on close
    cache_tools=True
)
```

### Agent Configuration

```python
agent = LLMAgent(
    jid="mcp-agent@example.com",
    password="password",
    provider=provider,
    system_prompt="You have access to external services via MCP tools. Use them when needed.",
    mcp_servers=[stdio_config, sse_config]
)
```


## Best Practices

### Server Development

- **Error Handling**: Always return proper error responses
- **Input Validation**: Validate all tool parameters
- **Resource Cleanup**: Properly close database connections
- **Logging**: Include detailed logging for debugging
- **Security**: Validate and sanitize all inputs

### Agent Configuration

- **Tool Caching**: Enable caching for better performance
- **Environment Variables**: Use env vars for configuration
- **Error Recovery**: Handle MCP server failures gracefully
- **Tool Selection**: Configure agents with relevant tools only

## Troubleshooting

### Common Issues

**Server not starting**:
- Check command and arguments
- Verify working directory
- Check environment variables

**Tool discovery fails**:
- Test server manually with JSON-RPC calls
- Check server implements required methods
- Verify JSON-RPC format

**Tool execution errors**:
- Check parameter schemas match
- Validate input data types
- Handle exceptions in tool functions


MCP integration provides powerful capabilities for connecting SPADE_LLM agents to external services and data sources. Start with simple STDIO servers and gradually build more complex integrations as needed.
