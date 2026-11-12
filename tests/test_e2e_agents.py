"""End to end: Zero hands a task to an agent, answers its question and gets the report.

A real service and WebSocket; the model is scripted, one script for the main
conversation and one for the agent (see ``RoleProvider``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.agents import AGENT_ID, RoleProvider
from tests.turns import text_of
from zero.config import Config
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import AppState, get_state

HELLO = {"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}}


@contextmanager
def service(
    provider: FakeProvider, config: Config | None = None
) -> Iterator[tuple[AppState, WebSocketTestSession]]:
    with TestClient(create_app(config or Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json(HELLO)
            assert ws.receive_json()["type"] == "hello.ok"
            yield state, ws


def call(tool: str, **tool_input: Any) -> list[Any]:
    return [ToolCallEvent(call=ToolCall(id=f"{tool}-1", name=tool, input=tool_input))]


def say(text: str) -> list[Any]:
    return [TextDelta(text=text)]


def send(ws: WebSocketTestSession, text: str, session_id: str | None = None) -> None:
    message = {"type": "message.send", "id": text, "text": text, "source": "desktop"}
    if session_id is not None:
        message["session_id"] = session_id
    ws.send_json(message)


def until_main_says(ws: WebSocketTestSession, text: str) -> list[dict[str, Any]]:
    """Every event up to the turn.finished whose text is ``text``."""
    seen = []
    while True:
        seen.append(event := ws.receive_json())
        if event["type"] == "turn.finished" and event["text"] == text:
            return seen


def test_an_agent_reads_asks_gets_the_answer_and_reports(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hello zero\nsecond line\n")
    config = Config()
    config.permissions.allowed_roots = [str(tmp_path)]
    provider = RoleProvider(
        main=FakeProvider(
            [
                call("agent_start", name="reader", task=f"Read {notes} and quote a line."),
                say("Started a reader."),
                call("agent_message", id=AGENT_ID, text="Quote the first line."),
                say("Answered the reader."),
                say("The reader says the first line is 'hello zero'."),
            ]
        ),
        agent=FakeProvider(
            [
                call("read_file", path=str(notes)),
                call("ask_parent", question="Which line should I quote?"),
                say("Asked; waiting for the answer."),
                say("The first line is 'hello zero'."),
            ]
        ),
    )
    with service(provider, config) as (state, ws):
        send(ws, "read my notes")
        seen = until_main_says(ws, "The reader says the first line is 'hello zero'.")
        agent_id = next(e["agent"]["id"] for e in seen if e["type"] == "agent.updated")

    updates = [e["agent"] for e in seen if e["type"] == "agent.updated"]
    assert [a["status"] for a in updates] == ["working", "asking", "working", "done"]
    assert updates[-1]["report"] == "The first line is 'hello zero'."
    main_session, agent_session = updates[0]["parent_session_id"], updates[0]["session_id"]

    # The agent read the file with its own tools.
    read = provider.agent.calls[1].messages[-1]
    assert read.role == "tool" and "hello zero" in text_of(read)

    # Its question reached the main conversation as a turn of its own...
    question = provider.main.calls[2].messages[-1]
    assert (question.role, text_of(question)) == (
        "user",
        f"Agent 'reader' (id {agent_id}) asks:\nWhich line should I quote?\n\n"
        "Answer it with agent_message, or ask the user first if only they know.",
    )
    # ...the answer came back to the agent as its next message...
    answer = provider.agent.calls[3].messages[-1]
    assert (answer.role, text_of(answer)) == ("user", "Quote the first line.")
    # ...and so did the report, to the main conversation.
    report = provider.main.calls[4].messages[-1]
    assert text_of(report) == (
        f"Agent 'reader' (id {agent_id}) has finished. Its report:\nThe first line is 'hello zero'."
    )

    started = [(e["session_id"], e["source"]) for e in seen if e["type"] == "turn.started"]
    assert [source for session, source in started if session == main_session] == [
        "desktop",
        "agent",
        "agent",
    ]
    assert [source for session, source in started if session == agent_session] == [
        "agent",
        "agent",
    ]
    of_agent = [e for e in seen if e.get("session_id") == agent_session and "turn_id" in e]
    assert all(e["agent"] == {"id": agent_id, "name": "reader"} for e in of_agent)
    assert provider.main.steps_left == provider.agent.steps_left == 0


def test_the_user_checks_on_an_agent_and_stops_it_midway() -> None:
    provider = RoleProvider(
        main=FakeProvider(
            [
                call("agent_start", name="digger", task="Dig through everything"),
                say("Started."),
                call("agent_status", id=AGENT_ID),
                say("It is still digging."),
                call("agent_stop", id=AGENT_ID),
                say("Stopped it."),
            ]
        ),
        agent=FakeProvider(infinite=True),
    )
    with service(provider) as (state, ws):
        send(ws, "dig")
        seen = until_main_says(ws, "Started.")
        main_session = seen[-1]["session_id"]
        agent = next(e["agent"] for e in seen if e["type"] == "agent.updated")

        send(ws, "how is it going?", main_session)
        until_main_says(ws, "It is still digging.")
        send(ws, "stop it", main_session)
        seen = until_main_says(ws, "Stopped it.")

        assert not state.turns.busy(agent["session_id"])

    status = text_of(provider.main.calls[3].messages[-1])
    assert status.startswith(f"Agent 'digger' (id {agent['id']}): working\n")
    stop_result = text_of(provider.main.calls[5].messages[-1])
    assert stop_result == "Agent 'digger' is stopped (stopped)."

    cancelled = next(
        e for e in seen if e["type"] == "turn.finished" and e["session_id"] == agent["session_id"]
    )
    assert cancelled["status"] == "cancelled"
    assert {"type": "session.deleted", "session_id": agent["session_id"]} in seen
    last = [e["agent"] for e in seen if e["type"] == "agent.updated"][-1]
    assert (last["status"], last.get("session_id")) == ("stopped", None)
