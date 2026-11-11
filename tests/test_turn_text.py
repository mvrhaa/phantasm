"""End-to-end tests for a text-only turn over the WebSocket."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from zero.config import Config
from zero.providers.base import LLMProvider, TextDelta, UsageEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import AppState, get_state

HELLO = {
    "type": "hello",
    "protocol": 1,
    "client": {
        "app": "desktop",
        "client_id": "c-1",
        "character": {
            "name": "Zero",
            "persona": "Curious.",
            "poses": ["idle"],
            "moods": ["neutral"],
        },
        "environment": "test client",
    },
}


@contextmanager
def running(
    provider: LLMProvider | None = None, config: Config | None = None
) -> Iterator[tuple[TestClient, AppState]]:
    with TestClient(create_app(config or Config(), provider)) as client:
        yield client, get_state(client.app)  # type: ignore[arg-type]


@contextmanager
def connected(
    client: TestClient, state: AppState, client_id: str = "c-1"
) -> Iterator[WebSocketTestSession]:
    with client.websocket_connect(f"/ws?token={state.token}") as ws:
        hello = {**HELLO, "client": {**HELLO["client"], "client_id": client_id}}  # type: ignore[dict-item]
        ws.send_json(hello)
        assert ws.receive_json()["type"] == "hello.ok"
        yield ws


def send(ws: WebSocketTestSession, text: str, **fields: Any) -> None:
    ws.send_json({"type": "message.send", "id": "m1", "text": text, "source": "desktop", **fields})


def until_finished(ws: WebSocketTestSession) -> list[dict[str, Any]]:
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] in ("turn.finished", "error"):
            return events


def titled(ws: WebSocketTestSession) -> dict[str, Any]:
    """Wait for the title a new session gets once its first turn is over."""
    event = ws.receive_json()
    assert event["type"] == "session.updated"
    return event


def history(ws: WebSocketTestSession, session_id: str) -> list[dict[str, Any]]:
    ws.send_json({"type": "history.get", "session_id": session_id})
    return ws.receive_json()["messages"]


def test_echo_turn_emits_events_in_order() -> None:
    with running() as (client, state), connected(client, state) as ws:
        send(ws, "hello big world")
        events = until_finished(ws)

    types = [event["type"] for event in events]
    assert types == [
        "session.created",
        "turn.started",
        *["text.delta"] * 4,
        "tool.started",
        "expression",
        "tool.finished",
        "turn.finished",
    ]

    started, finished = events[1], events[-1]
    session_id = events[0]["session"]["id"]
    assert started["reply_to"] == "m1"
    assert (started["session_id"], started["source"]) == (session_id, "desktop")
    assert all(e["turn_id"] == started["turn_id"] for e in events[1:])
    assert all(e["session_id"] == session_id for e in events[1:])
    assert "".join(e["delta"] for e in events[2:6]) == "echo: hello big world"
    assert (events[6]["tool"], events[6]["hidden"]) == ("set_expression", True)
    assert (events[7]["pose"], events[7]["mood"]) == ("idle", "neutral")
    assert finished["status"] == "ok"
    assert finished["text"] == "echo: hello big world"
    assert finished["expression"] == {"pose": "idle", "mood": "neutral"}
    assert finished["usage"] == {"input_tokens": 3, "output_tokens": 4}
    assert "error" not in finished


def test_answer_is_stored_in_history() -> None:
    with running() as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        events = until_finished(ws)
        session_id = events[0]["session"]["id"]
        turn_id = events[1]["turn_id"]
        titled(ws)

        stored = history(ws, session_id)

    assert [(m["role"], m["content"][0]["text"]) for m in stored] == [
        ("user", "hi"),
        ("assistant", "echo: hi"),
        ("tool", "Expression set."),
    ]
    assert all(m["turn_id"] == turn_id for m in stored)
    assert all((m["status"], m["source"]) == ("ok", "desktop") for m in stored)


def test_message_into_an_existing_session_creates_no_new_one() -> None:
    with running() as (client, state), connected(client, state) as ws:
        ws.send_json({"type": "session.create", "title": "Plans"})
        session_id = ws.receive_json()["session"]["id"]

        send(ws, "hi", session_id=session_id)
        events = until_finished(ws)

        ws.send_json({"type": "session.list"})
        sessions = ws.receive_json()["sessions"]

    assert events[0]["type"] == "turn.started"
    assert events[0]["session_id"] == session_id
    assert [s["id"] for s in sessions] == [session_id]


def test_unknown_session_gives_session_not_found() -> None:
    with running() as (client, state), connected(client, state) as ws:
        send(ws, "hi", session_id="missing")
        events = until_finished(ws)

    assert events == [
        {
            "type": "error",
            "code": "session_not_found",
            "message": "no session with id missing",
            "reply_to": "m1",
        }
    ]


def test_scripted_provider_text_reaches_the_client() -> None:
    provider = FakeProvider(
        [[TextDelta(text="Hel"), TextDelta(text="lo"), UsageEvent(input_tokens=7, output_tokens=2)]]
    )
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        events = until_finished(ws)

    assert [e["delta"] for e in events if e["type"] == "text.delta"] == ["Hel", "lo"]
    assert events[-1]["text"] == "Hello"
    assert events[-1]["usage"] == {"input_tokens": 7, "output_tokens": 2}


def test_model_receives_system_prompt_and_the_user_message() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "what time is it?", source="phone")
        until_finished(ws)

    (call,) = provider.calls
    system, user = call.messages
    assert call.model == "fake-1"
    assert [tool.name for tool in call.tools] == ["set_expression"]
    assert system.role == "system"
    system_text = system.content[0].text  # type: ignore[union-attr]
    assert system_text.startswith("# Who you are")
    assert "Your name is Zero." in system_text
    assert "cannot see the computer screen" in system_text
    assert "# Current time" in system_text
    assert (user.role, user.content[0].text) == ("user", "what time is it?")  # type: ignore[union-attr]


def test_second_turn_sends_the_earlier_conversation() -> None:
    provider = FakeProvider([[TextDelta(text="first answer")], [TextDelta(text="second answer")]])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "first question")
        session_id = until_finished(ws)[0]["session"]["id"]
        send(ws, "second question", session_id=session_id)
        until_finished(ws)

    second = provider.calls[1].messages
    assert [m.role for m in second] == ["system", "user", "assistant", "user"]
    assert [m.content[0].text for m in second[1:]] == [  # type: ignore[union-attr]
        "first question",
        "first answer",
        "second question",
    ]


def test_provider_error_ends_the_turn_with_status_error() -> None:
    provider = FakeProvider([[TextDelta(text="partial")]], fail_times=1, fail_retryable=False)
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        events = until_finished(ws)
        titled(ws)
        stored = history(ws, events[0]["session"]["id"])

        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

    finished = events[-1]
    assert [e["type"] for e in events] == ["session.created", "turn.started", "turn.finished"]
    assert (finished["status"], finished["text"]) == ("error", "")
    assert finished["error"] == "FakeProvider simulated failure"
    assert [(m["role"], m["status"]) for m in stored] == [("user", "ok"), ("assistant", "error")]


def test_unexpected_failure_ends_the_turn_without_leaking_details() -> None:
    class Exploding(FakeProvider):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            raise RuntimeError("secret detail")
            yield  # pragma: no cover

    with running(Exploding()) as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        finished = until_finished(ws)[-1]

    assert (finished["status"], finished["error"]) == ("error", "internal error")


def test_every_client_sees_the_turn() -> None:
    with running() as (client, state):
        with connected(client, state, "c-1") as first, connected(client, state, "c-2") as second:
            send(first, "hi")
            mine = until_finished(first)
            theirs = until_finished(second)

    assert [e["type"] for e in mine] == [e["type"] for e in theirs]
    assert mine[1]["reply_to"] == "m1"
    assert "reply_to" not in theirs[1]
    assert mine[-1] == theirs[-1]


def test_turn_updates_the_session_time() -> None:
    with running() as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        created = until_finished(ws)[0]["session"]
        titled(ws)

        ws.send_json({"type": "session.list"})
        (listed,) = ws.receive_json()["sessions"]

    assert listed["updated_at"] > created["updated_at"]


def test_finished_turn_is_forgotten_by_the_manager() -> None:
    with running() as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        until_finished(ws)
        client.portal.call(state.turns.wait_idle)

        assert state.turns.active == []


def test_unusable_provider_gives_provider_not_configured() -> None:
    config = Config()
    config.provider.name = "nope"
    with running(config=config) as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        events = until_finished(ws)

        ws.send_json({"type": "session.list"})
        sessions = ws.receive_json()["sessions"]

    assert [(e["type"], e["code"], e["reply_to"]) for e in events] == [
        ("error", "provider_not_configured", "m1")
    ]
    assert sessions == []


def test_shutdown_stops_a_running_turn() -> None:
    with running(FakeProvider(infinite=True)) as (client, state), connected(client, state) as ws:
        send(ws, "hi")
        while ws.receive_json()["type"] != "text.delta":
            pass
        assert len(state.turns.active) == 1


@pytest.mark.parametrize("text", ["", "Привет ☀️", "x" * 5000])
def test_any_text_is_echoed(text: str) -> None:
    with running() as (client, state), connected(client, state) as ws:
        send(ws, text)
        finished = until_finished(ws)[-1]

    assert finished["text"] == f"echo: {text}"
