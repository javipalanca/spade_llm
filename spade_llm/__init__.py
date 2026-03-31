"""SPADE_LLM - Extension de SPADE para integrar Large Language Models en agentes."""

from . import rag
from .agent import ChatAgent, CoordinatorAgent, LLMAgent, RetrievalAgent
from .behaviour import HumanInteractionBehaviour, LLMBehaviour, RetrievalBehaviour
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
from .structured_output import ReadyForStructuredOutputTool
from .tools import HumanInTheLoopTool, LangChainToolAdapter, LLMTool, RetrievalTool
from .utils import load_env_vars
from .version import __version__

__all__ = [
    "LLMBehaviour",
    "HumanInteractionBehaviour",
    "RetrievalBehaviour",
    "ContextManager",
    "LLMTool",
    "HumanInTheLoopTool",
    "LLMAgent",
    "ChatAgent",
    "CoordinatorAgent",
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
    "LangChainToolAdapter",
    # Structured output
    "ReadyForStructuredOutputTool",
    "__version__",
]
