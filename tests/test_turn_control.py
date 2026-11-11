"""Tests of retries, busy sessions and cancellation of turns."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.turns import FuncTool, Harness, RecordingClient, registry, roles, text_of, until
from zero.config import Config
from zero.core.turn import NOT_RUN, RETRY_DELAYS_S
from zero.core.turn_manager import TurnManager
from zero.protocol.messages import Envelope, TurnFinished
from zero.providers.base import ProviderError, StreamEvent, TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.base import ToolContext


class BreaksAfterText(FakeProvider):
    """Starts an answer, then fails with an error that would be worth retrying."""

    attempts = 0

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        self.attempts += 1
        yield TextDelta(text="Hel")
        raise ProviderError("connection reset", retryable=True)


# --- Retries -----------------------------------------------------------------


async def test_retryable_failure_is_retried(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="ok")]], fail_times=2)

    await harness.turn(provider).run()

    assert len(provider.calls) == 3
    finished = harness.finished
    assert (finished.status, finished.text, finished.error) == ("ok", "ok", None)
    assert harness.types == ["turn.started", "text.delta", "turn.finished"]


async def test_turn_gives_up_after_two_retries(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="ok")]], fail_times=3)

    await harness.turn(provider).run()

    assert len(provider.calls) == 3
    assert (harness.finished.status, harness.finished.error) == (
        "error",
        "FakeProvider simulated failure",
    )
    assert [(m.role, m.status) for m in await harness.stored()] == [
        ("user", "ok"),
        ("assistant", "error"),
    ]


async def test_non_retryable_failure_is_not_retried(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="ok")]], fail_times=1, fail_retryable=False)

    await harness.turn(provider).run()

    assert len(provider.calls) == 1
    assert harness.finished.status == "error"


async def test_failure_after_text_is_not_retried(harness: Harness) -> None:
    provider = BreaksAfterText()

    await harness.turn(provider).run()

    assert provider.attempts == 1
    finished = harness.finished
    assert (finished.status, finished.text, finished.error) == ("error", "Hel", "connection reset")
    answer = (await harness.stored())[-1]
    assert (answer.status, text_of(answer)) == ("error", "Hel")


async def test_retry_waits_and_the_wait_ends_on_cancel(harness: Harness) -> None:
    assert RETRY_DELAYS_S == (1.0, 4.0)
    provider = FakeProvider([[TextDelta(text="late")]], fail_times=1)
    turn = harness.turn(provider, retry_delays=RETRY_DELAYS_S)

    task = asyncio.create_task(turn.run())
    await until(lambda: len(provider.calls) == 1)
    await asyncio.sleep(0.05)
    assert not task.done()  # still pausing before the retry

    turn.cancel.cancel()
    async with asyncio.timeout(0.5):
        await task

    assert len(provider.calls) == 1
    assert (harness.finished.status, harness.finished.error) == ("cancelled", None)


# --- Cancellation ------------------------------------------------------------


async def test_cancel_mid_stream_keeps_the_partial_answer(harness: Harness) -> None:
    turn = harness.turn(FakeProvider(infinite=True))

    task = asyncio.create_task(turn.run())
    await until(lambda: len(harness.deltas) >= 3)
    turn.cancel.cancel()
    await task

    finished = harness.finished
    assert (finished.status, finished.error) == ("cancelled", None)
    assert finished.text == "".join(harness.deltas)
    assert finished.text.startswith("la la la ")
    user, answer = await harness.stored()
    assert (user.role, user.status) == ("user", "ok")
    assert (answer.role, answer.status, text_of(answer)) == (
        "assistant",
        "cancelled",
        finished.text,
    )


async def test_turn_cancelled_before_it_starts_never_calls_the_model(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="hi")]])
    turn = harness.turn(provider)

    turn.cancel.cancel()
    await turn.run()

    assert provider.calls == []
    assert harness.types == ["turn.started", "turn.finished"]
    assert (harness.finished.status, harness.finished.text) == ("cancelled", "")


async def test_cancel_during_tools_skips_the_calls_not_started(harness: Harness) -> None:
    ran: list[str] = []

    async def stop(params: Any, ctx: ToolContext) -> str:
        ran.append("stop")
        ctx.cancel.cancel()
        return "stopping"

    async def count(params: Any, ctx: ToolContext) -> str:
        ran.append("count")
        return "1"

    tools = registry(FuncTool("stop", stop), FuncTool("count", count))
    provider = FakeProvider(
        [
            [
                ToolCallEvent(call=ToolCall(id="c1", name="stop")),
                ToolCallEvent(call=ToolCall(id="c2", name="count")),
            ],
            [TextDelta(text="never")],
        ]
    )

    await harness.turn(provider, tools).run()

    assert ran == ["stop"]
    assert len(provider.calls) == 1
    assert harness.finished.status == "cancelled"
    assert [(e.type, e.call_id) for e in harness.tool_events()] == [
        ("tool.started", "c1"),
        ("tool.finished", "c1"),
    ]
    stored = await harness.stored()
    assert roles(stored) == "user assistant tool tool assistant"
    _, step, first, second, last = stored
    assert [call["id"] for call in step.tool_calls] == ["c1", "c2"]
    assert (first.tool_call_id, first.is_error, text_of(first)) == ("c1", False, "stopping")
    assert (second.tool_call_id, second.is_error, text_of(second)) == ("c2", True, NOT_RUN)
    assert (last.status, last.content) == ("cancelled", [])


# --- Busy sessions -----------------------------------------------------------


async def test_manager_knows_running_turns_and_busy_sessions(harness: Harness) -> None:
    manager = TurnManager()
    turn = harness.turn(FakeProvider(infinite=True))
    assert not manager.busy(harness.session_id)

    manager.start(turn)
    await until(lambda: len(harness.deltas) > 0)

    assert manager.busy(harness.session_id)
    assert not manager.busy("another-session")
    assert manager.get(turn.id) is turn

    turn.cancel.cancel()
    await manager.wait_idle()

    assert not manager.busy(harness.session_id)
    assert manager.get(turn.id) is None


async def test_session_is_free_once_clients_see_turn_finished(harness: Harness) -> None:
    manager = TurnManager()
    busy_at_finish: list[bool] = []

    class Watcher(RecordingClient):
        async def send(self, event: Envelope) -> bool:
            if isinstance(event, TurnFinished):
                busy_at_finish.append(manager.busy(harness.session_id))
            return await super().send(event)

    harness.client = Watcher()
    manager.start(harness.turn(FakeProvider([[TextDelta(text="hi")]])))
    await manager.wait_idle()

    assert busy_at_finish == [False]


async def test_work_after_a_turn_runs_once_turn_finished_is_out(harness: Harness) -> None:
    manager = TurnManager()
    seen: list[tuple[str, bool]] = []

    async def after() -> None:
        seen.append((harness.types[-1], manager.busy(harness.session_id)))

    manager.start(harness.turn(FakeProvider([[TextDelta(text="hi")]])), after=after)
    await manager.wait_idle()

    assert seen == [("turn.finished", False)]


async def test_failing_work_after_a_turn_is_logged(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    manager = TurnManager()

    async def after() -> None:
        raise RuntimeError("boom")

    turn = harness.turn(FakeProvider([[TextDelta(text="hi")]]))
    manager.start(turn, after=after)
    await manager.wait_idle()

    assert harness.finished.status == "ok"
    assert manager.active == []
    assert f"Follow-up of turn {turn.id} failed" in caplog.text


# --- Over the WebSocket ------------------------------------------------------


@contextmanager
def connected(provider: FakeProvider) -> Iterator[WebSocketTestSession]:
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            assert ws.receive_json()["type"] == "hello.ok"
            yield ws


def send(ws: WebSocketTestSession, message_id: str, **fields: Any) -> None:
    ws.send_json(
        {"type": "message.send", "id": message_id, "text": "sing", "source": "desktop", **fields}
    )


def read_until(ws: WebSocketTestSession, *types: str) -> list[dict[str, Any]]:
    """Read events up to and including the first one whose type is in ``types``."""
    events: list[dict[str, Any]] = []
    while not events or events[-1]["type"] not in types:
        events.append(ws.receive_json())
    return events


def test_cancel_over_websocket_saves_the_partial_answer() -> None:
    with connected(FakeProvider(infinite=True)) as ws:
        send(ws, "m1")
        events = read_until(ws, "text.delta")
        session_id, turn_id = events[0]["session"]["id"], events[1]["turn_id"]

        ws.send_json({"type": "turn.cancel", "id": "x1", "turn_id": turn_id})
        events += read_until(ws, "turn.finished")
        read_until(ws, "session.updated")

        ws.send_json({"type": "history.get", "session_id": session_id})
        stored = ws.receive_json()["messages"]

    finished = events[-1]
    deltas = [e["delta"] for e in events if e["type"] == "text.delta"]
    assert (finished["status"], finished["turn_id"]) == ("cancelled", turn_id)
    assert finished["text"] == "".join(deltas)
    assert "error" not in finished
    assert [m["status"] for m in stored] == ["ok", "cancelled"]
    assert stored[1]["content"] == [{"type": "text", "text": finished["text"]}]


def test_busy_session_refuses_a_message_until_its_turn_ends() -> None:
    with connected(FakeProvider(infinite=True)) as ws:
        send(ws, "m1")
        events = read_until(ws, "turn.started")
        session_id, turn_id = events[0]["session"]["id"], events[1]["turn_id"]

        send(ws, "m2", session_id=session_id)
        refused = read_until(ws, "error", "turn.started")[-1]
        ws.send_json({"type": "turn.cancel", "turn_id": turn_id})
        read_until(ws, "turn.finished")

        send(ws, "m3", session_id=session_id)
        accepted = read_until(ws, "error", "turn.started")[-1]
        ws.send_json({"type": "turn.cancel", "turn_id": accepted["turn_id"]})
        read_until(ws, "turn.finished")

    assert refused == {
        "type": "error",
        "code": "turn_busy",
        "message": "a turn is already running in this session",
        "reply_to": "m2",
    }
    assert (accepted["type"], accepted["reply_to"]) == ("turn.started", "m3")


def test_other_sessions_are_not_busy() -> None:
    with connected(FakeProvider(infinite=True)) as ws:
        send(ws, "m1")
        first = read_until(ws, "turn.started")[-1]

        send(ws, "m2")
        second = read_until(ws, "error", "turn.started")[-1]
        for started in (first, second):
            ws.send_json({"type": "turn.cancel", "turn_id": started["turn_id"]})
        read_until(ws, "turn.finished")
        read_until(ws, "turn.finished")

    assert (second["type"], second["reply_to"]) == ("turn.started", "m2")
    assert second["session_id"] != first["session_id"]


def test_cancel_of_an_unknown_turn_is_ignored() -> None:
    with connected(FakeProvider()) as ws:
        ws.send_json({"type": "turn.cancel", "id": "x1", "turn_id": "missing"})
        ws.send_json({"type": "ping", "id": "p1"})

        assert ws.receive_json() == {"type": "pong", "reply_to": "p1"}
