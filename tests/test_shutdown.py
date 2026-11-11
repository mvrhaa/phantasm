"""Tests of the graceful shutdown: turns end properly and clients are told why they go."""

import asyncio
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.turns import FuncTool, Harness, registry, until
from zero import __main__ as entry
from zero.approvals.manager import WITHDRAWN_BY
from zero.config import Config
from zero.core.turn_manager import TurnManager
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server import app as app_module
from zero.server.app import create_app, graceful_stop
from zero.server.state import AppState, get_state
from zero.server.ws import CLOSE_GOING_AWAY
from zero.tools.base import ToolContext

HELLO = {"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}}


# --- TurnManager.cancel_all ---------------------------------------------------


async def test_cancel_all_ends_running_turns_as_cancelled(harness: Harness) -> None:
    manager = TurnManager()
    manager.start(harness.turn(FakeProvider(infinite=True)))
    await until(lambda: len(harness.deltas) > 0)

    await manager.cancel_all(grace_s=2.0)

    assert manager.active == []
    assert harness.finished.status == "cancelled"
    assert [m.status for m in await harness.stored()] == ["ok", "cancelled"]


async def test_cancel_all_cuts_off_a_turn_that_does_not_stop(harness: Harness) -> None:
    started = asyncio.Event()

    async def stubborn(params: Any, ctx: ToolContext) -> str:
        started.set()
        await asyncio.sleep(30)  # never looks at ctx.cancel
        return "late"

    script = [[ToolCallEvent(call=ToolCall(id="s1", name="stubborn"))], [TextDelta(text="x")]]
    manager = TurnManager()
    manager.start(harness.turn(FakeProvider(script), registry(FuncTool("stubborn", stubborn))))
    async with asyncio.timeout(2):
        await started.wait()

    async with asyncio.timeout(2):
        await manager.cancel_all(grace_s=0.05)

    assert manager.active == []


async def test_cancel_all_without_turns_returns_at_once() -> None:
    async with asyncio.timeout(1):
        await TurnManager().cancel_all(grace_s=30)


# --- graceful_stop in the service --------------------------------------------


@contextmanager
def service(provider: FakeProvider) -> Iterator[tuple[TestClient, AppState]]:
    with TestClient(create_app(Config(), provider)) as client:
        yield client, get_state(client.app)  # type: ignore[arg-type]


@contextmanager
def connected(client: TestClient, state: AppState) -> Iterator[WebSocketTestSession]:
    with client.websocket_connect(f"/ws?token={state.token}") as ws:
        ws.send_json(HELLO)
        assert ws.receive_json()["type"] == "hello.ok"
        yield ws


def receive_until(ws: WebSocketTestSession, kind: str) -> dict[str, Any]:
    while True:
        event = ws.receive_json()
        if event["type"] == kind:
            return event


def stop(client: TestClient, state: AppState) -> None:
    client.portal.call(graceful_stop, state)  # type: ignore[union-attr]


def test_stop_withdraws_approvals_ends_turns_and_closes_clients() -> None:
    call = ToolCall(id="r1", name="run_command", input={"command": "echo hi"})
    provider = FakeProvider([[ToolCallEvent(call=call)], [TextDelta(text="ran")]])
    with service(provider) as (client, state), connected(client, state) as ws:
        ws.send_json({"type": "message.send", "id": "m1", "text": "run it", "source": "desktop"})
        request = receive_until(ws, "approval.request")

        stop(client, state)

        resolved = receive_until(ws, "approval.resolved")
        assert resolved["approval_id"] == request["approval_id"]
        assert (resolved["decision"], resolved["by_client_id"]) == ("deny", WITHDRAWN_BY)
        assert receive_until(ws, "turn.finished")["status"] == "cancelled"
        while (frame := ws.receive())["type"] != "websocket.close":
            pass  # the session's title may still arrive
        assert (frame["code"], frame["reason"]) == (CLOSE_GOING_AWAY, "Zero is shutting down")
        assert state.turns.active == []
        assert state.approvals.waiting == []


def test_stop_halts_the_scheduler_and_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    with service(FakeProvider([])) as (client, state):
        assert state.scheduler.running
        stop(client, state)
        assert not state.scheduler.running
        assert state.stopping

        cancelled: list[float] = []
        monkeypatch.setattr(state.turns, "cancel_all", lambda grace_s: cancelled.append(grace_s))
        stop(client, state)
        assert cancelled == []


def test_messages_are_refused_while_stopping() -> None:
    with service(FakeProvider([[TextDelta(text="hi")]])) as (client, state):
        with connected(client, state) as ws:
            state.stopping = True
            ws.send_json({"type": "message.send", "id": "m1", "text": "hi", "source": "desktop"})
            error = ws.receive_json()

        assert (error["type"], error["code"], error["reply_to"]) == ("error", "internal", "m1")
        assert "shutting down" in error["message"]
        assert state.turns.active == []


def test_new_clients_are_turned_away_while_stopping() -> None:
    with service(FakeProvider([])) as (client, state):
        state.stopping = True
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            closing = ws.receive()

        assert (closing["type"], closing["code"]) == ("websocket.close", CLOSE_GOING_AWAY)
        assert state.connections.connections == []


def test_leaving_the_app_stops_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    stops: list[AppState] = []
    original = app_module.graceful_stop

    async def recording(state: AppState, grace_s: float = app_module.SHUTDOWN_GRACE_S) -> None:
        stops.append(state)
        await original(state, grace_s)

    monkeypatch.setattr(app_module, "graceful_stop", recording)
    with service(FakeProvider([])) as (_, state):
        pass

    assert stops == [state]
    assert state.stopping


# --- uvicorn -----------------------------------------------------------------


async def test_server_winds_zero_down_before_uvicorn_drops_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def zero_stop(state: AppState) -> None:
        order.append("zero")

    async def uvicorn_stop(
        self: uvicorn.Server, sockets: list[socket.socket] | None = None
    ) -> None:
        order.append("uvicorn")

    monkeypatch.setattr(entry, "graceful_stop", zero_stop)
    monkeypatch.setattr(uvicorn.Server, "shutdown", uvicorn_stop)
    app = create_app(Config(), FakeProvider([]))
    server = entry.ZeroServer(uvicorn.Config(app, log_config=None))  # leave logging alone

    await server.shutdown()  # before startup there is no state: only uvicorn stops
    app.state.zero = object.__new__(AppState)
    await server.shutdown()

    assert order == ["uvicorn", "zero", "uvicorn"]
