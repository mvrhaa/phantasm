"""End to end: two clients, a client action that needs approval, and shared approvals.

The desktop declares ``screenshot`` and starts a turn; the phone watches. The
approval reaches both, the phone answers it, and the desktop's card closes
through ``approval.resolved``. Only the desktop gets the ``action.call``; its
picture reaches the model on the next step.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from zero.config import Config
from zero.protocol.messages import ImagePart
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state

PIXEL = {"type": "image", "mime": "image/png", "data_b64": "iVBORw0KGgo="}
SCREENSHOT = {
    "name": "screenshot",
    "description": "Capture the user's screen",
    "input_schema": {"type": "object", "properties": {"display": {"type": "integer"}}},
    "policy": "ask",
    "timeout_s": 5,
}


def read_until(ws: WebSocketTestSession, event_type: str) -> list[dict[str, Any]]:
    events = [ws.receive_json()]
    while events[-1]["type"] != event_type:
        events.append(ws.receive_json())
    return events


@pytest.fixture
def setup() -> Iterator[tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession]]:
    call = ToolCall(id="shot-1", name="screenshot", input={"display": 1})
    provider = FakeProvider([[ToolCallEvent(call=call)], [TextDelta(text="Nice wallpaper.")]])
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with (
            client.websocket_connect(f"/ws?token={token}") as desktop,
            client.websocket_connect(f"/ws?token={token}") as phone,
        ):
            for ws, client_id, actions in (
                (desktop, "desktop", [SCREENSHOT]),
                (phone, "phone", []),
            ):
                ws.send_json(
                    {
                        "type": "hello",
                        "protocol": 1,
                        "client": {"app": client_id, "client_id": client_id, "actions": actions},
                    }
                )
                assert ws.receive_json()["type"] == "hello.ok"
            yield provider, desktop, phone


def test_two_clients_share_the_turn_the_approval_and_the_action(
    setup: tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession],
) -> None:
    provider, desktop, phone = setup
    desktop.send_json({"type": "message.send", "id": "m1", "text": "look", "source": "desktop"})

    on_phone = read_until(phone, "approval.request")
    request = on_phone[-1]
    phone.send_json(
        {"type": "approval.respond", "approval_id": request["approval_id"], "decision": "allow"}
    )
    on_desktop = read_until(desktop, "action.call")
    action_call = on_desktop[-1]
    desktop.send_json(
        {"type": "action.result", "call_id": action_call["call_id"], "ok": True, "content": [PIXEL]}
    )
    on_desktop += read_until(desktop, "turn.finished")
    on_phone += read_until(phone, "turn.finished")

    # the approval closed on the desktop too, answered by the phone
    resolved = [e for e in on_desktop if e["type"] == "approval.resolved"]
    assert [(e["decision"], e["by_client_id"]) for e in resolved] == [("allow", "phone")]
    assert [e for e in on_phone if e["type"] == "approval.resolved"] == resolved

    # only the desktop was asked to act
    assert (action_call["action"], action_call["input"]) == ("screenshot", {"display": 1})
    assert not any(e["type"] == "action.call" for e in on_phone)

    # both saw the same turn, apart from the action call
    shared = [e for e in on_desktop if e["type"] != "action.call"]
    assert [e["type"] for e in shared] == [e["type"] for e in on_phone]
    assert [e["type"] for e in on_phone] == [
        "session.created",
        "turn.started",
        "tool.started",
        "approval.request",
        "approval.resolved",
        "tool.finished",
        "text.delta",
        "turn.finished",
    ]
    assert on_desktop[-1] == on_phone[-1]
    assert on_desktop[-1]["text"] == "Nice wallpaper."

    # the picture reached the model on the next step
    result = provider.calls[1].messages[-1]
    assert (result.role, result.tool_call_id) == ("tool", "shot-1")
    assert result.content == [ImagePart.model_validate(PIXEL)]


def test_the_phone_cannot_answer_the_desktops_action(
    setup: tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession],
) -> None:
    provider, desktop, phone = setup
    desktop.send_json({"type": "message.send", "text": "look", "source": "desktop"})
    request = read_until(desktop, "approval.request")[-1]
    desktop.send_json(
        {"type": "approval.respond", "approval_id": request["approval_id"], "decision": "allow"}
    )
    call_id = read_until(desktop, "action.call")[-1]["call_id"]

    fake = {"type": "image", "mime": "image/png", "data_b64": "ZmFrZQ=="}
    phone.send_json({"type": "action.result", "call_id": call_id, "ok": True, "content": [fake]})
    desktop.send_json({"type": "action.result", "call_id": call_id, "ok": True, "content": [PIXEL]})
    read_until(desktop, "turn.finished")

    assert provider.calls[1].messages[-1].content == [ImagePart.model_validate(PIXEL)]
