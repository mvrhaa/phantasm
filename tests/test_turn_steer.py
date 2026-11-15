"""Tests for steering a running turn: the user's words reach the model at its next step."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from tests.test_turn_text import connected, running, send
from tests.turns import FuncTool, Harness, registry, roles
from zero.core.turn import Turn
from zero.protocol.messages import TextPart
from zero.providers.base import (
    Message,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)
from zero.providers.fake import FakeProvider


def call(name: str) -> ToolCallEvent:
    return ToolCallEvent(call=ToolCall(id=f"c-{name}", name=name, input={}))


def said(messages: Sequence[Message]) -> list[str]:
    return [
        part.text
        for message in messages
        if message.role == "user"
        for part in message.content
        if isinstance(part, TextPart)
    ]


async def test_steering_during_a_tool_reaches_the_next_step(harness: Harness) -> None:
    turn: Turn | None = None

    async def look(params: Any, ctx: Any) -> str:
        assert turn is not None
        assert turn.steer("also check the other folder")
        return "looked"

    provider = FakeProvider([[call("look")], [TextDelta(text="Done, both folders.")]])
    turn = harness.turn(provider, registry(FuncTool("look", look)), text="check the folder")
    await turn.run()

    assert turn.finished is not None and turn.finished.status == "ok"
    second = provider.calls[1].messages
    assert said(second)[-1] == "also check the other folder"
    assert roles(second[1:]) == "user assistant tool user"
    stored = await harness.messages.list(harness.session_id)
    assert roles(stored) == "user assistant tool user assistant"
    assert stored[3].turn_id == turn.id
    steered = [event for event in harness.client.received if event.type == "turn.steered"]
    assert [event.text for event in steered] == ["also check the other folder"]  # type: ignore[attr-defined]


class SteersWhileAnswering(FakeProvider):
    """Steers the turn while its first answer streams."""

    turn: Turn | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: Any,
    ) -> AsyncIterator[StreamEvent]:
        first = not self.calls
        async for event in super().stream(
            messages, tools, model, max_tokens=max_tokens, cancel=cancel
        ):
            if first and self.turn is not None:
                self.turn.steer("in Russian, please")
                first = False
            yield event


async def test_steering_during_the_answer_asks_for_one_more_step(harness: Harness) -> None:
    provider = SteersWhileAnswering([[TextDelta(text="Hello!")], [TextDelta(text="Привет!")]])
    turn = harness.turn(provider)
    provider.turn = turn
    await turn.run()

    assert len(provider.calls) == 2
    second = provider.calls[1].messages
    assert [m.role for m in second[-2:]] == ["assistant", "user"]
    assert said(second)[-1] == "in Russian, please"
    stored = await harness.messages.list(harness.session_id)
    assert roles(stored) == "user assistant user assistant"
    assert turn.finished is not None and turn.finished.status == "ok"


async def test_a_finished_turn_takes_no_steering(harness: Harness) -> None:
    turn = await harness.run(FakeProvider([[TextDelta(text="ok")]]))
    assert turn.steer("too late") is False


def test_steering_over_the_websocket() -> None:
    provider = FakeProvider(
        [[TextDelta(text="long answer")], [TextDelta(text="short")]], delay_s=0.3
    )
    with running(provider) as (client, state), connected(client, state) as ws:
        ws.send_json({"type": "turn.steer", "id": "s0", "turn_id": "nope", "text": "x"})
        error = ws.receive_json()
        assert (error["type"], error["code"], error["reply_to"]) == (
            "error",
            "turn_not_running",
            "s0",
        )

        send(ws, "write a long answer")
        started = ws.receive_json()
        while started["type"] != "turn.started":
            started = ws.receive_json()
        ws.send_json(
            {"type": "turn.steer", "id": "s1", "turn_id": started["turn_id"], "text": "shorter"}
        )
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "turn.finished":
                break

    steered = [event for event in events if event["type"] == "turn.steered"]
    assert steered == [
        {
            "type": "turn.steered",
            "reply_to": "s1",
            "session_id": started["session_id"],
            "turn_id": started["turn_id"],
            "text": "shorter",
        }
    ]
    assert said(provider.calls[-1].messages)[-1] == "shorter"
