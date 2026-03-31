"""Base classes and types for the guardrails system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class GuardrailAction(Enum):
    """Actions that a guardrail can take."""

    PASS = "pass"
    MODIFY = "modify"
    BLOCK = "block"
    WARNING = "warning"


@dataclass
class GuardrailResult:
    """Result of applying a guardrail."""

    action: GuardrailAction
    content: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_message: Optional[str] = None


class Guardrail(ABC):
    """Abstract base class for all guardrails."""

    def __init__(self, name: str, enabled: bool = True, blocked_message: Optional[str] = None):
        """
        Initialize a guardrail.

        Args:
            name: Name of the guardrail
            enabled: Whether the guardrail is active
            blocked_message: Custom message to return when content is blocked
        """
        self.name = name
        self.enabled = enabled
        self.blocked_message = blocked_message
        self._logger = logging.getLogger(f"spade_llm.guardrails.{name}")

    @abstractmethod
    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """
        Check the content and return the result.

        Args:
            content: The content to check
            context: Additional context information

        Returns:
            GuardrailResult indicating the action to take
        """
        pass

    async def __call__(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Allow using the guardrail as a function."""
        if not self.enabled:
            return GuardrailResult(action=GuardrailAction.PASS, content=content)

        result = await self.check(content, context)

        if result.action == GuardrailAction.BLOCK and self.blocked_message:
            result.custom_message = self.blocked_message

        return result
