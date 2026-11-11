"""Tests of the round trip of a client action: action.call out, action.result back."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.turns import text_of, until
from zero.config import Config
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.protocol.messages import (
    ActionCall,
    ActionDecl,
    ActionResult,
    Envelope,
    ImagePart,
)
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.base import ToolContext, ToolResult
from zero.tools.client_actions import ActionBroker
from zero.tools.permissions import PathGuard

SCREENSHOT = ActionDecl(name="screenshot", description="Capture the screen", policy="allow")
PIXEL = ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")


@dataclass
class FakeConnection:
    alive: bool = True
    sent: list[Envelope] = field(default_factory=list)

    async def send(self, event: Envelope) -> bool:
        if self.alive:
            self.sent.append(event)
        return self.alive


@dataclass
class FakeConnections:
    by_id: dict[str, FakeConnection]

    def get(self, connection_id: str) -> FakeConnection | None:
        return self.by_id.get(connection_id)


def context(cancel: CancelToken | None = None) -> ToolContext:
    config = Config()
    return ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="desk",
        cancel=cancel or CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )


def broker_with(**connections: FakeConnection) -> ActionBroker:
    return ActionBroker(FakeConnections(dict(connections)))


async def sent_call(connection: FakeConnection) -> ActionCall:
    await until(lambda: any(isinstance(e, ActionCall) for e in connection.sent))
    return next(e for e in connection.sent if isinstance(e, ActionCall))


def start(
    broker: ActionBroker,
    action: ActionDecl = SCREENSHOT,
    params: dict[str, Any] | None = None,
    cancel: CancelToken | None = None,
) -> asyncio.Task[ToolResult]:
    return asyncio.create_task(broker.call("desk", action, params or {}, context(cancel)))


# --- ActionBroker ------------------------------------------------------------


async def test_action_goes_only_to_its_client_and_the_answer_comes_back() -> None:
    desk, phone = FakeConnection(), FakeConnection()
    broker = broker_with(desk=desk, phone=phone)
    running = start(broker, params={"display": 1})

    call = await sent_call(desk)
    assert broker.resolve("desk", ActionResult(call_id=call.call_id, ok=True, content=[PIXEL]))
    result = await running

    assert (call.session_id, call.turn_id, call.action, call.input) == (
        "s1",
        "t1",
        "screenshot",
        {"display": 1},
    )
    assert phone.sent == []
    assert (result.content, result.is_error) == ([PIXEL], False)
    assert broker.waiting == 0


async def test_empty_success_says_done() -> None:
    desk = FakeConnection()
    broker = broker_with(desk=desk)
    running = start(broker)

    broker.resolve("desk", ActionResult(call_id=(await sent_call(desk)).call_id, ok=True))

    assert text_of(await running) == "screenshot done."


async def test_failure_on_the_client_is_an_error_for_the_model() -> None:
    desk = FakeConnection()
    broker = broker_with(desk=desk)
    running = start(broker)

    answer = ActionResult(call_id=(await sent_call(desk)).call_id, ok=False, error="screen locked")
    broker.resolve("desk", answer)
    result = await running

    assert (result.is_error, text_of(result)) == (
        True,
        "screenshot failed on the client: screen locked",
    )


async def test_only_the_client_that_got_the_call_can_answer() -> None:
    desk = FakeConnection()
    broker = broker_with(desk=desk, phone=FakeConnection())
    running = start(broker)
    call_id = (await sent_call(desk)).call_id

    assert not broker.resolve("phone", ActionResult(call_id=call_id, ok=True, content=[PIXEL]))
    assert broker.resolve("desk", ActionResult(call_id=call_id, ok=True))
    assert not broker.resolve("desk", ActionResult(call_id=call_id, ok=True))  # a repeat
    assert (await running).content != [PIXEL]


async def test_unknown_answer_is_ignored() -> None:
    assert not broker_with().resolve("desk", ActionResult(call_id="nope", ok=True))


async def test_silent_client_times_out() -> None:
    quick = SCREENSHOT.model_copy(update={"timeout_s": 0.1})

    result = await start(broker_with(desk=FakeConnection()), action=quick)

    assert (result.is_error, text_of(result)) == (
        True,
        "screenshot did not answer within 0.1 s (action_timeout).",
    )


@pytest.mark.parametrize(
    "connections", [{}, {"desk": FakeConnection(alive=False)}], ids=["gone", "dead"]
)
async def test_missing_client_is_client_gone(connections: dict[str, FakeConnection]) -> None:
    result = await start(broker_with(**connections))

    assert text_of(result) == (
        "screenshot could not run: the client that offers it has disconnected (client_gone)."
    )


async def test_client_leaving_while_the_action_runs_is_client_gone() -> None:
    desk = FakeConnection()
    broker = broker_with(desk=desk)
    running = start(broker)
    await sent_call(desk)

    broker.connection_closed("desk")

    assert "(client_gone)" in text_of(await running)
    assert broker.waiting == 0


async def test_cancel_stops_waiting() -> None:
    desk = FakeConnection()
    cancel = CancelToken()
    broker = broker_with(desk=desk)
    running = start(broker, cancel=cancel)
    await sent_call(desk)

    cancel.cancel()

    with pytest.raises(TurnCancelled):
        await running
    assert broker.waiting == 0


# --- Over the WebSocket ------------------------------------------------------


def hello(ws: WebSocketTestSession, client_id: str, actions: list[dict[str, Any]]) -> None:
    ws.send_json(
        {
            "type": "hello",
            "protocol": 1,
            "client": {"app": "t", "client_id": client_id, "actions": actions},
        }
    )
    assert ws.receive_json()["type"] == "hello.ok"


def read_until(ws: WebSocketTestSession, *types: str) -> list[dict[str, Any]]:
    events = [ws.receive_json()]
    while events[-1]["type"] not in types:
        events.append(ws.receive_json())
    return events


SCREENSHOT_DECL = {"name": "screenshot", "description": "Capture the screen", "policy": "allow"}


def script() -> list[list[Any]]:
    call = ToolCall(id="s1", name="screenshot", input={"display": 1})
    return [[ToolCallEvent(call=call)], [TextDelta(text="I see your screen.")]]


def test_screenshot_round_trip_reaches_the_model() -> None:
    provider = FakeProvider(script())
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as desk:
            hello(desk, "desk", [SCREENSHOT_DECL])
            desk.send_json(
                {"type": "message.send", "text": "what is on my screen?", "source": "desktop"}
            )
            call = read_until(desk, "action.call")[-1]
            desk.send_json(
                {
                    "type": "action.result",
                    "call_id": call["call_id"],
                    "ok": True,
                    "content": [PIXEL.model_dump()],
                }
            )
            events = read_until(desk, "turn.finished")

    assert (call["action"], call["input"]) == ("screenshot", {"display": 1})
    result = provider.calls[1].messages[-1]
    assert result.content == [PIXEL]
    assert [e["status"] for e in events if e["type"] == "tool.finished"] == ["ok"]
    assert events[-1]["text"] == "I see your screen."


def test_client_leaving_mid_action_gives_the_model_client_gone() -> None:
    provider = FakeProvider(script())
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        token = state.token
        with client.websocket_connect(f"/ws?token={token}") as watcher:
            hello(watcher, "phone", [])
            with client.websocket_connect(f"/ws?token={token}") as desk:
                hello(desk, "desk", [SCREENSHOT_DECL])
                desk.send_json({"type": "message.send", "text": "look", "source": "desktop"})
                read_until(desk, "action.call")
            events = read_until(watcher, "turn.finished")

    result = provider.calls[1].messages[-1]
    assert result.is_error
    assert text_of(result).endswith("(client_gone).")
    assert [e["status"] for e in events if e["type"] == "tool.finished"] == ["error"]
    assert events[-1]["status"] == "ok"
