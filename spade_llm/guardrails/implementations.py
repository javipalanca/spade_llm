"""Concrete implementations of guardrails."""

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from .base import GuardrailAction, GuardrailResult
from .types import InputGuardrail, OutputGuardrail

if TYPE_CHECKING:
    from ..providers.base_provider import LLMProvider


class KeywordGuardrail(InputGuardrail):
    """Blocks or modifies messages containing prohibited keywords."""

    def __init__(
        self,
        name: str,
        blocked_keywords: List[str],
        action: GuardrailAction = GuardrailAction.BLOCK,
        replacement: str = "[REDACTED]",
        case_sensitive: bool = False,
        enabled: bool = True,
        blocked_message: Optional[str] = None,
    ):
        """
        Initialize a keyword guardrail.

        Args:
            name: Name of the guardrail
            blocked_keywords: List of keywords to block
            action: Action to take (BLOCK or MODIFY)
            replacement: Text to replace blocked keywords with (if action is MODIFY)
            case_sensitive: Whether keyword matching is case sensitive
            enabled: Whether the guardrail is active
            blocked_message: Custom message when content is blocked
        """
        super().__init__(name, enabled, blocked_message)
        self.blocked_keywords = blocked_keywords if case_sensitive else [kw.lower() for kw in blocked_keywords]
        self.action = action
        self.replacement = replacement
        self.case_sensitive = case_sensitive

    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check for blocked keywords."""
        check_content = content if self.case_sensitive else content.lower()

        for keyword in self.blocked_keywords:
            if keyword in check_content:
                if self.action == GuardrailAction.BLOCK:
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        reason=f"Blocked keyword found: {keyword}",
                        custom_message=self.blocked_message,
                    )
                elif self.action == GuardrailAction.MODIFY:
                    # Replace the keyword
                    if self.case_sensitive:
                        modified = content.replace(keyword, self.replacement)
                    else:
                        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                        modified = pattern.sub(self.replacement, content)

                    return GuardrailResult(
                        action=GuardrailAction.MODIFY,
                        content=modified,
                        reason=f"Keyword '{keyword}' replaced",
                    )

        return GuardrailResult(action=GuardrailAction.PASS, content=content)


class LLMGuardrail(OutputGuardrail):
    """Uses a smaller LLM model to verify content safety."""

    def __init__(
        self,
        name: str,
        provider: "LLMProvider",
        safety_prompt: Optional[str] = None,
        enabled: bool = True,
        blocked_message: Optional[str] = None,
    ):
        """
        Initialize an LLM-based guardrail.

        Args:
            name: Name of the guardrail
            provider: LLM provider to use for safety checks
            safety_prompt: Custom prompt for safety checking
            enabled: Whether the guardrail is active
            blocked_message: Custom message when content is blocked
        """
        super().__init__(name, enabled, blocked_message)
        self.provider = provider
        self.safety_prompt = (
            safety_prompt
            or """
        Analyze the following text for safety and appropriateness.
        Respond with JSON: {"safe": true/false, "reason": "explanation if unsafe"}

        Text: {content}
        """
        )

    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check content safety using LLM."""
        # Create a temporary context for the safety model
        from ..context import ContextManager

        temp_context = ContextManager()
        temp_context.add_message_dict(
            {"role": "user", "content": self.safety_prompt.format(content=content)},
            "safety_check",
        )

        try:
            response = await self.provider.get_response(temp_context)

            # Parse JSON response
            result = json.loads(response)

            if not result.get("safe", True):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=result.get("reason", "Content deemed unsafe"),
                    custom_message=self.blocked_message,
                )
        except Exception as e:
            self._logger.warning(f"Error in LLM safety check: {e}. Assuming content is safe.")

        return GuardrailResult(action=GuardrailAction.PASS, content=content)


class RegexGuardrail(InputGuardrail):
    """Applies regex rules to detect and modify patterns."""

    def __init__(
        self,
        name: str,
        patterns: Dict[str, Union[str, GuardrailAction]],
        enabled: bool = True,
        blocked_message: Optional[str] = None,
    ):
        """
        Initialize a regex guardrail.

        Args:
            name: Name of the guardrail
            patterns: Dictionary of {regex_pattern: replacement_string or GuardrailAction}
            enabled: Whether the guardrail is active
            blocked_message: Custom message when content is blocked
        """
        super().__init__(name, enabled, blocked_message)
        self.patterns = patterns

    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Apply regex patterns to check content."""
        modified_content = content
        modifications_made = False

        for pattern, action_or_replacement in self.patterns.items():
            if re.search(pattern, content):
                if isinstance(action_or_replacement, GuardrailAction):
                    if action_or_replacement == GuardrailAction.BLOCK:
                        return GuardrailResult(
                            action=GuardrailAction.BLOCK,
                            reason=f"Pattern '{pattern}' detected",
                            custom_message=self.blocked_message,
                        )
                else:
                    # It's a replacement string
                    modified_content = re.sub(pattern, action_or_replacement, modified_content)
                    modifications_made = True

        if modifications_made:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                content=modified_content,
                reason="Patterns replaced",
            )

        return GuardrailResult(action=GuardrailAction.PASS, content=content)


class CustomFunctionGuardrail(InputGuardrail):
    """Allows using a custom function as a guardrail."""

    def __init__(
        self,
        name: str,
        check_function: Callable[[str, Dict[str, Any]], GuardrailResult],
        enabled: bool = True,
        blocked_message: Optional[str] = None,
    ):
        """
        Initialize a custom function guardrail.

        Args:
            name: Name of the guardrail
            check_function: Function that checks content and returns GuardrailResult
            enabled: Whether the guardrail is active
            blocked_message: Custom message when content is blocked
        """
        super().__init__(name, enabled, blocked_message)
        self.check_function = check_function

    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Execute the custom check function."""
        if asyncio.iscoroutinefunction(self.check_function):
            result = await self.check_function(content, context)
        else:
            result = await asyncio.to_thread(self.check_function, content, context)

        # Apply custom message if blocking and not already set
        if result.action == GuardrailAction.BLOCK and self.blocked_message and not result.custom_message:
            result.custom_message = self.blocked_message

        return result
