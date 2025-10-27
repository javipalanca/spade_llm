"""SPADE_LLM - Extension de SPADE para integrar Large Language Models en agentes."""

from .agent import ChatAgent, LLMAgent, RetrievalAgent
from .behaviour import HumanInteractionBehaviour, LLMBehaviour
from .context import ContextManager
from .guardrails import (
    CompositeGuardrail,
    CustomFunctionGuardrail,
    Guardrail,
    GuardrailAction,
    GuardrailResult,
    InputGuardrail,
    KeywordGuardrail,
    LLMGuardrail,
    OutputGuardrail,
    RegexGuardrail,
)
from .memory import AgentInteractionMemory, AgentMemoryTool
from .providers import LLMProvider
from .routing import RoutingFunction, RoutingResponse
from .tools import HumanInTheLoopTool, LLMTool, RetrievalTool
from .utils import load_env_vars
from .version import __version__
from . import rag

__all__ = [
    "LLMBehaviour",
    "HumanInteractionBehaviour",
    "ContextManager",
    "LLMTool",
    "HumanInTheLoopTool",
    "LLMAgent",
    "ChatAgent",
    "LLMProvider",
    "load_env_vars",
    "RoutingFunction",
    "RoutingResponse",
    # Memory
    "AgentInteractionMemory",
    "AgentMemoryTool",
    # Guardrails
    "Guardrail",
    "GuardrailAction",
    "GuardrailResult",
    "InputGuardrail",
    "OutputGuardrail",
    "CompositeGuardrail",
    "KeywordGuardrail",
    "LLMGuardrail",
    "RegexGuardrail",
    "CustomFunctionGuardrail",
    # RAG system
    "rag",
    "RetrievalAgent",
    "RetrievalTool",
    "__version__",
]
