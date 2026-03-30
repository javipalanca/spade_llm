# Structured Output

Generate LLM responses that conform to a predefined Pydantic schema, ensuring type-safe, parseable output from your agents.

## Overview

Structured output lets you define a Pydantic `BaseModel` as the expected response format. SPADE_LLM uses a **two-phase pattern** when tools are also present:

1. **Information Gathering** — The LLM uses regular tools to collect data.
2. **Structured Generation** — The LLM signals readiness and the system switches to the provider's parsing API for constrained generation.

When no tools are registered, the agent directly uses the structured output API.

## Quick Start

### Without Tools

```python
from pydantic import BaseModel
from spade_llm import LLMAgent, LLMProvider

class WeatherReport(BaseModel):
    city: str
    temperature: float
    summary: str

provider = LLMProvider(model="gpt-4o-mini")

agent = LLMAgent(
    jid="weather@example.com",
    password="password",
    provider=provider,
    system_prompt="You are a weather reporter. Always report the weather.",
    output_schema=WeatherReport
)
```

The agent's responses will be JSON strings conforming to `WeatherReport`.

### With Tools (Two-Phase Pattern)

When both `output_schema` and `tools` are specified, the system automatically injects a `ready_for_structured_output` signal tool:

```python
from pydantic import BaseModel
from spade_llm import LLMAgent, LLMProvider, LLMTool

class TripPlan(BaseModel):
    destination: str
    days: int
    budget: float
    activities: list[str]

async def search_flights(destination: str) -> str:
    return f"Flights to {destination}: $200-$500"

async def search_hotels(destination: str, days: int) -> str:
    return f"Hotels in {destination} for {days} days: $80-$200/night"

flight_tool = LLMTool(
    name="search_flights",
    description="Search for flights to a destination",
    parameters={
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Destination city"}
        },
        "required": ["destination"]
    },
    func=search_flights
)

hotel_tool = LLMTool(
    name="search_hotels",
    description="Search for hotels at a destination",
    parameters={
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Destination city"},
            "days": {"type": "integer", "description": "Number of days"}
        },
        "required": ["destination", "days"]
    },
    func=search_hotels
)

provider = LLMProvider(model="gpt-4o-mini")

agent = LLMAgent(
    jid="planner@example.com",
    password="password",
    provider=provider,
    system_prompt="You are a trip planner. Use the tools to research, then provide a structured plan.",
    tools=[flight_tool, hotel_tool],
    output_schema=TripPlan
)
```

**Flow:**

1. LLM receives the user's request along with `search_flights`, `search_hotels`, and the auto-injected `ready_for_structured_output` tool.
2. LLM calls `search_flights` and `search_hotels` to gather data.
3. LLM calls `ready_for_structured_output` to signal it has enough context.
4. System switches to the parsing API and generates a `TripPlan` response.

## How It Works

### Without Tools

When `output_schema` is set and no tools are registered:

- The provider passes the schema directly to the LLM API via `response_format`.
- The LLM generates a response constrained to the schema.
- The response is returned as a JSON string.

### With Tools (Two-Phase Pattern)

When both `output_schema` and tools are set:

1. A `ReadyForStructuredOutputTool` is automatically injected into the tool list.
2. The LLM can use regular tools for information gathering.
3. When the LLM has enough context, it calls `ready_for_structured_output`.
4. The system makes a second LLM call with `output_schema` and **no tools**, producing the structured response.

This pattern avoids issues that arise when mixing tool calling with structured output in a single LLM call.

## Per-Conversation Schemas

You can also set output schemas per conversation via the `ContextManager`:

```python
from spade_llm.context import ContextManager

context = ContextManager(system_prompt="You are a helpful assistant.")
context.set_output_schema(WeatherReport, conversation_id="conv_1")
context.set_output_schema(TripPlan, conversation_id="conv_2")
```

Per-conversation schemas take precedence over the agent-level `output_schema`.

## Response Format

Structured responses are serialized as JSON strings before being sent as XMPP messages. The receiving agent can parse them back:

```python
import json
from pydantic import BaseModel

# On the receiving side
data = json.loads(message.body)
report = WeatherReport(**data)
print(report.city, report.temperature)
```

## Design Rationale

This implementation follows a similar approach to Google's ADK (Agent Development Kit):

- **Signal-based phase switching** avoids provider issues when mixing tools with structured output in a single call.
- **No tools in the structured phase** reduces LLM confusion and improves schema compliance.
- **Reuses existing tool infrastructure** — the ready signal is just another tool, requiring no special protocol changes.

## Limitations

- Structured output support depends on the underlying LLM provider. Models that support OpenAI-compatible `response_format` (GPT-4o, GPT-4o-mini, etc.) work best.
- Output guardrails are skipped for structured responses since the schema already constrains the output.
