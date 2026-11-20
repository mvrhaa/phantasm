"""A message with ``command.shell``: the turn runs the command itself, without the model."""

from typing import Any

import pytest
from pydantic import ValidationError

from phantasm.protocol.messages import MessageCommand
from phantasm.providers.fake import FakeProvider
from tests.test_turn_text import connected, history, running, send


def read_until(ws: Any, event_type: str) -> list[dict[str, Any]]:
    events = [ws.receive_json()]
    while events[-1]["type"] != event_type:
        events.append(ws.receive_json())
    return events


def answer(ws: Any, decision: str) -> list[dict[str, Any]]:
    request = read_until(ws, "approval.request")[-1]
    assert (request["tool"], request["label"]) == ("run_command", "$ echo hello")
    ws.send_json(
        {"type": "approval.respond", "approval_id": request["approval_id"], "decision": decision}
    )
    return read_until(ws, "turn.finished")


def test_the_command_runs_once_allowed_and_its_output_is_the_answer() -> None:
    provider = FakeProvider([])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "echo hello", command={"name": "Shell", "shell": "echo hello"})
        events = answer(ws, "allow")
        session_id = events[-1]["session_id"]
        titled = read_until(ws, "session.updated")[-1]
        messages = history(ws, session_id)

    assert provider.calls == []
    assert events[-1]["status"] == "ok"
    assert events[-1]["text"] == "```\nhello\n```"
    deltas = [event["delta"] for event in events if event["type"] == "text.delta"]
    assert deltas == ["```\nhello\n```"]
    tools = [event for event in events if event["type"] == "tool.finished"]
    assert [tool["status"] for tool in tools] == ["ok"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[0]["command"] == "Shell"
    assert titled["session"]["title"] == "$ echo hello"


def test_a_denied_command_says_so_without_running() -> None:
    provider = FakeProvider([])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "echo hello", command={"name": "Shell", "shell": "echo hello"})
        events = answer(ws, "deny")

    assert provider.calls == []
    assert events[-1]["status"] == "ok"
    assert "```" not in events[-1]["text"]
    assert events[-1]["text"]


def test_an_empty_shell_command_is_refused() -> None:
    with pytest.raises(ValidationError):
        MessageCommand(name="Shell", shell="")
