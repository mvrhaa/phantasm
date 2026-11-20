"""A stand-in for a real LLM provider, used in development and all tests.

With a script it plays the script back. Without one it runs in echo mode:
it answers "echo: <the user's text>" and, if the ``set_expression`` tool is
offered, calls it once.
"""

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from phantasm.protocol.messages import TextPart
from phantasm.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)

EXPRESSION_TOOL = "set_expression"


@dataclass
class RecordedCall:
    """What one ``stream`` call received, kept so tests can inspect it."""

    messages: list[Message]
    tools: list[ToolSpec]
    model: str
    max_tokens: int


def _text_of(message: Message) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def _first_choice(tool: ToolSpec, field: str) -> str | None:
    """Return the first allowed value of an enum field in a tool's JSON Schema."""
    choices = tool.input_schema.get("properties", {}).get(field, {}).get("enum")
    return str(choices[0]) if choices else None


class FakeProvider:
    """Plays back a script, or echoes the user when there is none.

    ``script`` is a list of steps; each ``stream`` call yields the next step.
    A step that does not end with a ``StopEvent`` gets one added: ``tool_use``
    if the step called a tool, ``end`` otherwise.

    The other options imitate a misbehaving provider:

    - ``delay_s``: pause before every event;
    - ``fail_times``: the first N calls raise ``ProviderError``
      (``fail_retryable`` says of which kind) without using up a script step;
    - ``infinite``: stream text forever, until the turn is cancelled.
    """

    name = "fake"
    writes_prose = False

    def __init__(
        self,
        script: Sequence[Sequence[StreamEvent]] | None = None,
        *,
        delay_s: float = 0.0,
        fail_times: int = 0,
        fail_retryable: bool = True,
        infinite: bool = False,
    ) -> None:
        self._script = [list(step) for step in script] if script is not None else None
        self._played = 0
        self._failures_left = fail_times
        self.delay_s = delay_s
        self.fail_retryable = fail_retryable
        self.infinite = infinite
        self.calls: list[RecordedCall] = []

    @property
    def steps_left(self) -> int:
        """How many scripted steps have not been played yet (0 in echo mode)."""
        return len(self._script) - self._played if self._script is not None else 0

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        """Record the request and yield the answer.

        Cancellation is checked before every event, so a cancelled turn stops
        in the middle of a step with ``TurnCancelled``.
        """
        self.calls.append(
            RecordedCall(
                messages=[message.model_copy(deep=True) for message in messages],
                tools=[tool.model_copy(deep=True) for tool in tools],
                model=model,
                max_tokens=max_tokens,
            )
        )
        if self._failures_left > 0:
            self._failures_left -= 1
            raise ProviderError("FakeProvider simulated failure", retryable=self.fail_retryable)

        if self.infinite:
            events = self._forever()
        elif self._script is not None:
            events = self._next_step()
        else:
            events = self._echo(messages, tools)

        async for event in events:
            if self.delay_s > 0:
                await asyncio.sleep(self.delay_s)
            cancel.raise_if_cancelled()
            yield event
            await asyncio.sleep(0)

    async def _next_step(self) -> AsyncIterator[StreamEvent]:
        assert self._script is not None
        if self._played >= len(self._script):
            raise ProviderError(f"FakeProvider script has no step {self._played + 1}")
        events = self._script[self._played]
        self._played += 1
        for event in events:
            yield event
        if not events or not isinstance(events[-1], StopEvent):
            called_tool = any(isinstance(event, ToolCallEvent) for event in events)
            yield StopEvent(reason="tool_use" if called_tool else "end")

    async def _echo(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[StreamEvent]:
        """Answer the last user message; after a tool result just finish the turn."""
        if messages and messages[-1].role == "tool":
            yield StopEvent(reason="end")
            return

        user_text = next((_text_of(m) for m in reversed(messages) if m.role == "user"), "")
        reply = f"echo: {user_text}"
        for chunk in re.findall(r"\S+\s*|\s+", reply):
            yield TextDelta(text=chunk)
        yield UsageEvent(input_tokens=len(user_text.split()), output_tokens=len(reply.split()))

        expression = next((tool for tool in tools if tool.name == EXPRESSION_TOOL), None)
        pose = _first_choice(expression, "pose") if expression else None
        mood = _first_choice(expression, "mood") if expression else None
        if pose is not None and mood is not None:
            call = ToolCall(
                id=f"fake-call-{len(self.calls)}",
                name=EXPRESSION_TOOL,
                input={"pose": pose, "mood": mood},
            )
            yield ToolCallEvent(call=call)
            yield StopEvent(reason="tool_use")
        else:
            yield StopEvent(reason="end")

    async def _forever(self) -> AsyncIterator[StreamEvent]:
        while True:
            yield TextDelta(text="la ")

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        """The fake cannot count tokens; callers fall back to their own estimate."""
        return None

    def supports_images(self, model: str) -> bool:
        """The fake accepts images so tests can check that they reach the model."""
        return True
