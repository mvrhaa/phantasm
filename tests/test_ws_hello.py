"""Tests for the hello handshake."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

import zero
from zero.config import Config
from zero.server.app import create_app
from zero.server.state import AppState, get_state

HELLO: dict[str, Any] = {
    "type": "hello",
    "id": "h1",
    "protocol": 1,
    "client": {
        "app": "desktop",
        "client_id": "c-1",
        "character": {"name": "Zero", "poses": ["idle"], "moods": ["neutral"]},
        "actions": [{"name": "screenshot", "description": "Capture the screen"}],
    },
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Config())) as test_client:
        yield test_client


@pytest.fixture
def state(client: TestClient) -> AppState:
    return get_state(client.app)  # type: ignore[arg-type]


@pytest.fixture
def ws(client: TestClient, state: AppState) -> Iterator[WebSocketTestSession]:
    with client.websocket_connect(f"/ws?token={state.token}") as session:
        yield session


def test_hello_is_answered_with_hello_ok(ws: WebSocketTestSession) -> None:
    ws.send_json(HELLO)

    assert ws.receive_json() == {
        "type": "hello.ok",
        "reply_to": "h1",
        "zero_version": zero.__version__,
        "protocol": 1,
        "sessions": [],
    }


def test_hello_ok_lists_sessions_newest_first(
    client: TestClient, state: AppState, ws: WebSocketTestSession
) -> None:
    older = client.portal.call(state.sessions.create, "older")
    newer = client.portal.call(state.sessions.create, "newer")
    client.portal.call(state.sessions.touch, newer.id)

    ws.send_json(HELLO)
    sessions = ws.receive_json()["sessions"]

    assert [s["id"] for s in sessions] == [newer.id, older.id]
    assert [s["title"] for s in sessions] == ["newer", "older"]
    assert set(sessions[0]) == {"id", "title", "updated_at"}


def test_hello_stores_the_declaration_on_the_connection(
    state: AppState, ws: WebSocketTestSession
) -> None:
    ws.send_json(HELLO)
    ws.receive_json()

    (connection,) = state.connections.connections
    assert connection.client_id == "c-1"
    assert connection.hello is not None
    assert connection.hello.character is not None
    assert connection.hello.character.name == "Zero"
    assert [action.name for action in connection.hello.actions] == ["screenshot"]


def test_repeated_hello_replaces_the_declaration(state: AppState, ws: WebSocketTestSession) -> None:
    ws.send_json(HELLO)
    ws.receive_json()
    second = {
        "type": "hello",
        "id": "h2",
        "protocol": 1,
        "client": {"app": "desktop", "client_id": "c-1", "environment": "updated"},
    }

    ws.send_json(second)
    reply = ws.receive_json()

    (connection,) = state.connections.connections
    assert (reply["type"], reply["reply_to"]) == ("hello.ok", "h2")
    assert connection.hello is not None
    assert connection.hello.environment == "updated"
    assert connection.hello.character is None
    assert connection.hello.actions == []


@pytest.mark.parametrize("version", [0, 2, 99])
def test_other_protocol_version_gets_mismatch_and_is_disconnected(
    version: int, state: AppState, ws: WebSocketTestSession
) -> None:
    ws.send_json({**HELLO, "protocol": version})

    error = ws.receive_json()
    assert (error["type"], error["code"], error["reply_to"]) == ("error", "protocol_mismatch", "h1")
    with pytest.raises(WebSocketDisconnect) as excinfo:
        ws.receive_json()
    assert excinfo.value.code == 4400
    assert state.connections.connections == []


def test_ping_works_before_hello(ws: WebSocketTestSession) -> None:
    ws.send_json({"type": "ping", "id": "p1"})

    assert ws.receive_json() == {"type": "pong", "reply_to": "p1"}


@pytest.mark.parametrize(
    "message",
    [
        {"type": "session.list"},
        {"type": "session.create", "title": "Plans"},
        {"type": "message.send", "text": "hi", "source": "desktop"},
        {"type": "history.get", "session_id": "s1"},
        {"type": "turn.cancel", "turn_id": "t1"},
    ],
    ids=lambda message: message["type"],
)
def test_other_messages_before_hello_are_refused(
    message: dict[str, Any], state: AppState, client: TestClient, ws: WebSocketTestSession
) -> None:
    ws.send_json({**message, "id": "m1"})

    assert ws.receive_json() == {
        "type": "error",
        "code": "invalid_message",
        "message": "send hello before anything else",
        "reply_to": "m1",
    }
    assert client.portal.call(state.sessions.list) == []

    ws.send_json(HELLO)
    assert ws.receive_json()["type"] == "hello.ok"


def test_invalid_hello_keeps_the_connection_open(ws: WebSocketTestSession) -> None:
    ws.send_json({"type": "hello", "id": "h1", "protocol": 1})

    error = ws.receive_json()
    assert (error["code"], error["reply_to"]) == ("invalid_message", "h1")

    ws.send_json(HELLO)
    assert ws.receive_json()["type"] == "hello.ok"


def test_each_connection_has_its_own_declaration(client: TestClient, state: AppState) -> None:
    url = f"/ws?token={state.token}"
    with client.websocket_connect(url) as first, client.websocket_connect(url) as second:
        first.send_json(HELLO)
        first.receive_json()
        second.send_json({**HELLO, "client": {"app": "phone", "client_id": "c-2"}})
        second.receive_json()

        assert sorted(c.client_id or "" for c in state.connections.connections) == ["c-1", "c-2"]
