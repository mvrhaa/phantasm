"""A temporary tool registry for the agent loop; commit 35 replaces it with the real one."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from zero.protocol.messages import ContentPart, TextPart
from zero.providers.base import ToolCall, ToolSpec

logger = logging.getLogger(__name__)

StubToolFn = Callable[[dict[str, Any]], Awaitable[str]]
"""A stub tool: takes the call's input and returns text for the model."""


@dataclass
class StubToolResult:
    """What a tool call gave back to the model."""

    content: list[ContentPart]
    is_error: bool = False


def error_result(text: str) -> StubToolResult:
    """A result that tells the model the call did not work."""
    return StubToolResult(content=[TextPart(text=text)], is_error=True)


class StubToolRegistry:
    """Tools the model may call, kept as plain async functions.

    Failures never escape ``execute``: an unknown tool or an exception in a
    tool becomes an error result, so the model can read it and try again.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, StubToolFn]] = {}

    def register(
        self,
        name: str,
        fn: StubToolFn,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        """Add a tool. Raises ``ValueError`` if the name is already taken."""
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
        )
        self._tools[name] = (spec, fn)

    def specs(self) -> list[ToolSpec]:
        """Describe every tool for the model."""
        return [spec for spec, _ in self._tools.values()]

    async def execute(self, call: ToolCall) -> StubToolResult:
        """Run a call and return its result; errors come back as results too."""
        entry = self._tools.get(call.name)
        if entry is None:
            available = ", ".join(self._tools) or "none"
            return error_result(f"Unknown tool {call.name!r}. Available tools: {available}.")
        _, fn = entry
        try:
            output = await fn(call.input)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", call.name, type(exc).__name__)
            return error_result(f"Tool {call.name!r} failed: {str(exc) or type(exc).__name__}")
        return StubToolResult(content=[TextPart(text=output)])
