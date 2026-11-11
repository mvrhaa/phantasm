"""A scripted stand-in for a real LLM provider, used in development and all tests."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from zero.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    StopEvent,
    StreamEvent,
    ToolCallEvent,
    ToolSpec,
)


@dataclass
class RecordedCall:
    """What one ``stream`` call received, kept so tests can inspect it."""

    messages: list[Message]
    tools: list[ToolSpec]
    model: str
    max_tokens: int


class FakeProvider:
    """Plays back a script: each ``stream`` call yields the next step's events.

    A step that does not end with a ``StopEvent`` gets one added: ``tool_use``
    if the step called a tool, ``end`` otherwise.
    """

    name = "fake"

    def __init__(self, script: Sequence[Sequence[StreamEvent]] | None = None) -> None:
        self._script = [list(step) for step in script or []]
        self.calls: list[RecordedCall] = []

    @property
    def steps_left(self) -> int:
        """How many scripted steps have not been played yet."""
        return len(self._script) - len(self.calls)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        """Record the request and yield the next scripted step.

        Cancellation is checked before every event, so a cancelled turn stops
        in the middle of a step with ``TurnCancelled``.
        """
        step_index = len(self.calls)
        self.calls.append(
            RecordedCall(
                messages=[message.model_copy(deep=True) for message in messages],
                tools=[tool.model_copy(deep=True) for tool in tools],
                model=model,
                max_tokens=max_tokens,
            )
        )
        if step_index >= len(self._script):
            raise ProviderError(f"FakeProvider script has no step {step_index + 1}")

        events = list(self._script[step_index])
        if not events or not isinstance(events[-1], StopEvent):
            called_tool = any(isinstance(event, ToolCallEvent) for event in events)
            events.append(StopEvent(reason="tool_use" if called_tool else "end"))

        for event in events:
            cancel.raise_if_cancelled()
            yield event
            await asyncio.sleep(0)  # let other tasks run, as a real network stream would

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        """The fake cannot count tokens; callers fall back to their own estimate."""
        return None

    def supports_images(self, model: str) -> bool:
        """The fake accepts images so tests can check that they reach the model."""
        return True
