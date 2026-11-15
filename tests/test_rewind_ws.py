"""Tests for session.rewind over the WebSocket."""

from typing import Any

from starlette.testclient import WebSocketTestSession

from tests.test_turn_text import connected, running, send, until_finished
from zero.providers.base import TextDelta
from zero.providers.fake import FakeProvider


def history(ws: WebSocketTestSession, session_id: str) -> list[dict[str, Any]]:
    ws.send_json({"type": "history.get", "id": "h", "session_id": session_id})
    while True:
        event = ws.receive_json()
        if event["type"] == "history.ok":
            return event["messages"]


def test_rewind_over_the_websocket() -> None:
    provider = FakeProvider([[TextDelta(text="a1")], [TextDelta(text="a2")]])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "first")
        events = until_finished(ws)
        session_id = events[-1]["session_id"]
        ws.receive_json()  # the session's new title
        send(ws, "second", session_id=session_id)
        until_finished(ws)
        second = [m for m in history(ws, session_id) if m["role"] == "user"][-1]["id"]

        ws.send_json(
            {
                "type": "session.rewind",
                "id": "r0",
                "session_id": session_id,
                "message_id": second,
                "dry_run": True,
            }
        )
        dry = ws.receive_json()
        assert len(history(ws, session_id)) == 4

        ws.send_json(
            {"type": "session.rewind", "id": "r1", "session_id": session_id, "message_id": second}
        )
        done = ws.receive_json()
        left = history(ws, session_id)

        ws.send_json(
            {"type": "session.rewind", "id": "r2", "session_id": "nope", "message_id": second}
        )
        missing = ws.receive_json()

    assert (dry["type"], dry["reply_to"], dry["dry_run"], dry["removed_messages"]) == (
        "session.rewound",
        "r0",
        True,
        2,
    )
    assert (done["reply_to"], done["dry_run"], done["text"], done["removed_messages"]) == (
        "r1",
        False,
        "second",
        2,
    )
    assert [m["role"] for m in left] == ["user", "assistant"]
    assert (missing["type"], missing["code"], missing["reply_to"]) == (
        "error",
        "session_not_found",
        "r2",
    )
