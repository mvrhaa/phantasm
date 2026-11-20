"""A stand-in for a real LLM provider, used in development and all tests.

With a script it plays the script back. Without one it runs in echo mode:
it answers "echo: <the user's text>" and, if the ``set_expression`` tool is
offered, calls it once. A message "call <tool> <JSON input>" makes it call that
tool instead, and then say what the tool returned, so a client can check a whole
round trip through a real brain.
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from phantasm.protocol.messages import ContentPart, FilePart, ImagePart, TextPart
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
_CALL = re.compile(r"call (\S+)(?:\s+(.*))?", re.DOTALL)


@dataclass
class RecordedCall:
    """What one ``stream`` call received, kept so tests can inspect it."""

    messages: list[Message]
    tools: list[ToolSpec]
    model: str
    max_tokens: int


def _text_of(message: Message) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def _named_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """The tool and input of a "call <tool> <JSON input>" message, or None."""
    match = _CALL.fullmatch(text.strip())
    if match is None:
        return None
    try:
        params = json.loads(match.group(2)) if match.group(2) else {}
    except json.JSONDecodeError:
        return None
    return (match.group(1), params) if isinstance(params, dict) else None


def _describe(part: ContentPart) -> str:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, ImagePart):
        return f"[image {part.mime}]"
    if isinstance(part, FilePart):
        return f"[file {part.name} {part.mime or '?'}]"
    return f"[{part.type}]"


def _tool_report(messages: list[Message]) -> str:
    """What the tools called last returned, one line each; set_expression says nothing."""
    results = []
    for message in reversed(messages):
        if message.role != "tool":
            break
        results.append(message)
    asked = next((m for m in reversed(messages) if m.role == "assistant"), None)
    names = {call.id: call.name for call in asked.tool_calls} if asked else {}
    lines = []
    for result in reversed(results):
        name = names.get(result.tool_call_id or "", "tool")
        if name == EXPRESSION_TOOL:
            continue
        said = " ".join(_describe(part) for part in result.content)
        lines.append(f"{name} failed: {said}" if result.is_error else f"{name}: {said}")
    return "\n".join(lines)


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
        """Answer the last user message, or call the tool it names; after a tool result say
        what a named tool returned and finish the turn."""
        if messages and messages[-1].role == "tool":
            report = _tool_report(messages)
            if report:
                yield TextDelta(text=report)
            yield StopEvent(reason="end")
            return

        user_text = next((_text_of(m) for m in reversed(messages) if m.role == "user"), "")
        named = _named_call(user_text)
        if named is not None and any(tool.name == named[0] for tool in tools):
            name, params = named
            yield ToolCallEvent(
                call=ToolCall(id=f"fake-call-{len(self.calls)}", name=name, input=params)
            )
            yield StopEvent(reason="tool_use")
            return
        reply = f"echo: no tool {named[0]}" if named is not None else f"echo: {user_text}"
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
