"""Tests for the WebSocket read loop and message dispatch."""

from collections.abc import Iterator
from typing import Any, get_args

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from zero.config import Config
from zero.protocol.messages import ClientMessage, Ping
from zero.server import handlers
from zero.server.app import create_app
from zero.server.state import get_state


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Config())) as test_client:
        yield test_client


@pytest.fixture
def ws(client: TestClient) -> Iterator[WebSocketTestSession]:
    token = get_state(client.app).token  # type: ignore[arg-type]
    with client.websocket_connect("/ws", headers={"Authorization": f"Bearer {token}"}) as session:
        yield session


def ping(ws: WebSocketTestSession, message_id: str = "p") -> dict[str, Any]:
    ws.send_json({"type": "ping", "id": message_id})
    return ws.receive_json()


def test_ping_is_answered_with_pong(ws: WebSocketTestSession) -> None:
    assert ping(ws, "p1") == {"type": "pong", "reply_to": "p1"}


def test_ping_without_id_gets_pong_without_reply_to(ws: WebSocketTestSession) -> None:
    ws.send_json({"type": "ping"})

    assert ws.receive_json() == {"type": "pong"}


@pytest.mark.parametrize("raw", ["{not json", "", "[]", '"hello"', "42"])
def test_garbage_gives_invalid_message_and_keeps_the_connection(
    raw: str, ws: WebSocketTestSession
) -> None:
    ws.send_text(raw)

    error = ws.receive_json()
    assert error["type"] == "error"
    assert error["code"] == "invalid_message"
    assert "reply_to" not in error
    assert ping(ws)["type"] == "pong"


def test_unknown_type_gives_unknown_type_and_keeps_the_connection(
    ws: WebSocketTestSession,
) -> None:
    ws.send_json({"type": "session.explode", "id": "m1"})

    assert ws.receive_json() == {
        "type": "error",
        "code": "unknown_type",
        "message": "unknown message type",
        "reply_to": "m1",
    }
    assert ping(ws)["type"] == "pong"


def test_missing_type_gives_invalid_message(ws: WebSocketTestSession) -> None:
    ws.send_json({"id": "m1", "text": "hi"})

    error = ws.receive_json()
    assert (error["code"], error["reply_to"]) == ("invalid_message", "m1")


def test_invalid_fields_give_invalid_message_naming_the_field(ws: WebSocketTestSession) -> None:
    ws.send_json({"type": "session.rename", "id": "m1", "session_id": "s1", "title": 42})

    error = ws.receive_json()
    assert (error["code"], error["reply_to"]) == ("invalid_message", "m1")
    assert "title" in error["message"]
    assert "42" not in error["message"]


def test_binary_frame_gives_invalid_message(ws: WebSocketTestSession) -> None:
    ws.send_bytes(b"\x00\x01")

    assert ws.receive_json()["code"] == "invalid_message"
    assert ping(ws)["type"] == "pong"


def test_failing_handler_gives_internal_error_and_keeps_the_connection(
    ws: WebSocketTestSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(connection: Any, message: Any, state: Any) -> None:
        raise RuntimeError("secret detail")

    monkeypatch.setitem(handlers.HANDLERS, Ping, broken)
    ws.send_json({"type": "ping", "id": "p1"})

    assert ws.receive_json() == {
        "type": "error",
        "code": "internal",
        "message": "internal error",
        "reply_to": "p1",
    }
    monkeypatch.undo()
    assert ping(ws)["type"] == "pong"


def test_message_without_a_handler_is_answered_with_an_error(
    ws: WebSocketTestSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(handlers.HANDLERS, Ping)
    ws.send_json({"type": "ping", "id": "m1"})

    error = ws.receive_json()
    assert (error["type"], error["reply_to"]) == ("error", "m1")
    assert error["message"] == "ping is not supported yet"


def test_every_client_message_has_a_handler() -> None:
    message_types = set(get_args(get_args(ClientMessage)[0]))

    assert message_types == set(handlers.HANDLERS)


def test_connection_is_registered_while_open(client: TestClient) -> None:
    manager = get_state(client.app).connections  # type: ignore[arg-type]
    token = get_state(client.app).token  # type: ignore[arg-type]
    assert manager.connections == []

    with client.websocket_connect(f"/ws?token={token}") as first:
        ping(first)
        with client.websocket_connect(f"/ws?token={token}") as second:
            ping(second)
            assert len(manager.connections) == 2
        ping(first)
        assert len(manager.connections) == 1

    client.get("/health")
    assert manager.connections == []


def test_rejected_connection_is_never_registered(client: TestClient) -> None:
    manager = get_state(client.app).connections  # type: ignore[arg-type]

    with client.websocket_connect("/ws") as rejected:
        with pytest.raises(Exception):  # noqa: B017
            rejected.receive_text()

    assert manager.connections == []
