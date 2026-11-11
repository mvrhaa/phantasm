"""End-to-end tests for session and history handlers over the WebSocket."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from zero.config import Config
from zero.protocol.messages import TextPart
from zero.server.app import create_app
from zero.server.state import AppState, get_state


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Config())) as test_client:
        yield test_client


@pytest.fixture
def state(client: TestClient) -> AppState:
    return get_state(client.app)  # type: ignore[arg-type]


def connect(client: TestClient, state: AppState, client_id: str) -> Any:
    return client.websocket_connect(f"/ws?token={state.token}")


def hello(ws: WebSocketTestSession, client_id: str = "c-1") -> dict[str, Any]:
    ws.send_json(
        {"type": "hello", "protocol": 1, "client": {"app": "test", "client_id": client_id}}
    )
    return ws.receive_json()


def ask(ws: WebSocketTestSession, message: dict[str, Any]) -> dict[str, Any]:
    ws.send_json(message)
    return ws.receive_json()


@pytest.fixture
def ws(client: TestClient, state: AppState) -> Iterator[WebSocketTestSession]:
    with connect(client, state, "c-1") as session:
        hello(session)
        yield session


def test_hello_create_rename_list_delete(ws: WebSocketTestSession) -> None:
    created = ask(ws, {"type": "session.create", "id": "m1", "title": "Plans"})
    session_id = created["session"]["id"]
    assert (created["type"], created["reply_to"]) == ("session.created", "m1")
    assert created["session"]["title"] == "Plans"

    renamed = ask(
        ws, {"type": "session.rename", "id": "m2", "session_id": session_id, "title": "Summer"}
    )
    assert (renamed["type"], renamed["reply_to"]) == ("session.updated", "m2")
    assert renamed["session"] == {**created["session"], **renamed["session"], "title": "Summer"}

    listed = ask(ws, {"type": "session.list", "id": "m3"})
    assert (listed["type"], listed["reply_to"]) == ("session.list.ok", "m3")
    assert [(s["id"], s["title"]) for s in listed["sessions"]] == [(session_id, "Summer")]

    deleted = ask(ws, {"type": "session.delete", "id": "m4", "session_id": session_id})
    assert deleted == {"type": "session.deleted", "reply_to": "m4", "session_id": session_id}

    assert ask(ws, {"type": "session.list"})["sessions"] == []


def test_session_without_title(ws: WebSocketTestSession) -> None:
    created = ask(ws, {"type": "session.create"})

    assert created["type"] == "session.created"
    assert "title" not in created["session"]


def test_second_client_sees_session_changes(client: TestClient, state: AppState) -> None:
    with connect(client, state, "c-1") as first, connect(client, state, "c-2") as second:
        hello(first, "c-1")
        hello(second, "c-2")

        created = ask(first, {"type": "session.create", "id": "m1", "title": "Plans"})
        session_id = created["session"]["id"]
        seen_created = second.receive_json()

        ask(first, {"type": "session.rename", "id": "m2", "session_id": session_id, "title": "New"})
        seen_updated = second.receive_json()

        ask(first, {"type": "session.delete", "id": "m3", "session_id": session_id})
        seen_deleted = second.receive_json()

    assert seen_created == {"type": "session.created", "session": created["session"]}
    assert seen_updated["type"] == "session.updated"
    assert seen_updated["session"]["title"] == "New"
    assert "reply_to" not in seen_updated
    assert seen_deleted == {"type": "session.deleted", "session_id": session_id}


def test_session_list_is_not_broadcast(client: TestClient, state: AppState) -> None:
    with connect(client, state, "c-1") as first, connect(client, state, "c-2") as second:
        hello(first, "c-1")
        hello(second, "c-2")

        ask(first, {"type": "session.list"})

        assert ask(second, {"type": "ping", "id": "p1"}) == {"type": "pong", "reply_to": "p1"}


@pytest.mark.parametrize(
    "message",
    [
        {"type": "session.rename", "session_id": "missing", "title": "New"},
        {"type": "session.delete", "session_id": "missing"},
        {"type": "history.get", "session_id": "missing"},
    ],
    ids=lambda message: message["type"],
)
def test_unknown_session_gives_session_not_found(
    message: dict[str, Any], client: TestClient, state: AppState
) -> None:
    with connect(client, state, "c-1") as first, connect(client, state, "c-2") as second:
        hello(first, "c-1")
        hello(second, "c-2")

        error = ask(first, {**message, "id": "m1"})

        assert (error["type"], error["code"], error["reply_to"]) == (
            "error",
            "session_not_found",
            "m1",
        )
        assert ask(second, {"type": "ping"}) == {"type": "pong"}


def test_history_of_a_new_session_is_empty(ws: WebSocketTestSession) -> None:
    session_id = ask(ws, {"type": "session.create"})["session"]["id"]

    assert ask(ws, {"type": "history.get", "id": "m1", "session_id": session_id}) == {
        "type": "history.ok",
        "reply_to": "m1",
        "session_id": session_id,
        "messages": [],
        "tool_runs": [],
    }


def add_messages(client: TestClient, state: AppState, session_id: str, count: int) -> None:
    for number in range(1, count + 1):
        client.portal.call(
            lambda number=number: state.messages.append(
                session_id,
                "user",
                [TextPart(text=f"message {number}")],
                status="ok",
                source="desktop",
            )
        )


def texts(history: dict[str, Any]) -> list[str]:
    return [message["content"][0]["text"] for message in history["messages"]]


def test_history_returns_messages_oldest_first(
    client: TestClient, state: AppState, ws: WebSocketTestSession
) -> None:
    session_id = ask(ws, {"type": "session.create"})["session"]["id"]
    add_messages(client, state, session_id, 3)

    history = ask(ws, {"type": "history.get", "session_id": session_id})

    assert texts(history) == ["message 1", "message 2", "message 3"]
    first = history["messages"][0]
    assert (first["role"], first["status"], first["source"]) == ("user", "ok", "desktop")
    assert first["content"] == [{"type": "text", "text": "message 1"}]
    assert {"id", "created_at"} <= set(first)


def test_history_pages_back_with_before_and_limit(
    client: TestClient, state: AppState, ws: WebSocketTestSession
) -> None:
    session_id = ask(ws, {"type": "session.create"})["session"]["id"]
    add_messages(client, state, session_id, 5)

    newest = ask(ws, {"type": "history.get", "session_id": session_id, "limit": 2})
    older = ask(
        ws,
        {
            "type": "history.get",
            "session_id": session_id,
            "limit": 2,
            "before": newest["messages"][0]["id"],
        },
    )

    assert texts(newest) == ["message 4", "message 5"]
    assert texts(older) == ["message 2", "message 3"]


def test_history_can_leave_out_tool_messages(
    client: TestClient, state: AppState, ws: WebSocketTestSession
) -> None:
    session_id = ask(ws, {"type": "session.create"})["session"]["id"]
    client.portal.call(lambda: state.messages.append(session_id, "user", [TextPart(text="hi")]))
    client.portal.call(
        lambda: state.messages.append(
            session_id, "tool", [TextPart(text="result")], tool_call_id="c1"
        )
    )

    with_tools = ask(ws, {"type": "history.get", "session_id": session_id})
    without_tools = ask(
        ws, {"type": "history.get", "session_id": session_id, "include_tools": False}
    )

    assert [m["role"] for m in with_tools["messages"]] == ["user", "tool"]
    assert [m["role"] for m in without_tools["messages"]] == ["user"]


def test_deleting_a_session_removes_its_history(
    client: TestClient, state: AppState, ws: WebSocketTestSession
) -> None:
    session_id = ask(ws, {"type": "session.create"})["session"]["id"]
    add_messages(client, state, session_id, 2)

    ask(ws, {"type": "session.delete", "session_id": session_id})

    assert ask(ws, {"type": "history.get", "session_id": session_id})["code"] == (
        "session_not_found"
    )
