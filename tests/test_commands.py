"""Tests for a command sent with a message: instructions, tools and model for one turn."""

import base64
from datetime import UTC, datetime
from typing import Any

from phantasm.config import Config
from phantasm.core.context import build_system_prompt
from phantasm.protocol.messages import ImagePart, TextPart
from phantasm.providers.base import TextDelta
from phantasm.providers.fake import FakeProvider
from tests.test_turn_text import HELLO, connected, history, running, send, until_finished

QUICK = {
    "name": "Quick",
    "instructions": "Answer in one sentence.",
    "tools": "none",
    "model": "small",
}
PIXEL = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()


def small_config() -> Config:
    config = Config()
    config.provider.small_model = "fake-small"
    return config


def system_text(call: Any) -> str:
    return call.messages[0].content[0].text


def test_command_instructions_join_the_system_prompt_before_the_time() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "what is 2+2?", command=QUICK)
        until_finished(ws)

    text = system_text(provider.calls[0])
    assert "# Command\nAnswer in one sentence." in text
    assert text.index("# Command") < text.index("# Current time")


def test_command_instructions_are_not_part_of_the_users_message() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "what is 2+2?", command=QUICK)
        until_finished(ws)

    user = provider.calls[0].messages[-1]
    assert [part.text for part in user.content if isinstance(part, TextPart)] == ["what is 2+2?"]


def test_tools_none_gives_the_model_no_tools() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "hi", command=QUICK)
        until_finished(ws)

    assert provider.calls[0].tools == []


def test_small_model_runs_the_turn_on_the_small_model() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "hi", command=QUICK)
        until_finished(ws)

    assert provider.calls[0].model == "fake-small"


def test_command_defaults_keep_the_tools_and_the_main_model() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "hi", command={"name": "Plain", "instructions": "Be kind."})
        until_finished(ws)

    (call,) = provider.calls
    assert call.model == "fake-1"
    assert "read_file" in [tool.name for tool in call.tools]
    assert "# Command\nBe kind." in system_text(call)


def test_the_next_message_without_a_command_has_no_command_block() -> None:
    provider = FakeProvider([[TextDelta(text="one")], [TextDelta(text="two")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "first", command=QUICK)
        session_id = until_finished(ws)[0]["session"]["id"]
        send(ws, "second", session_id=session_id)
        until_finished(ws)

    assert "# Command" not in system_text(provider.calls[1])
    assert provider.calls[1].model == "fake-1"


def test_history_keeps_the_command_name_with_the_users_message() -> None:
    provider = FakeProvider([[TextDelta(text="four")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "what is 2+2?", command=QUICK)
        session_id = until_finished(ws)[0]["session"]["id"]
        assert ws.receive_json()["type"] == "session.updated"
        messages = history(ws, session_id)

    assert [(m["role"], m.get("command")) for m in messages] == [
        ("user", "Quick"),
        ("assistant", None),
    ]


def test_empty_text_with_an_image_sends_only_the_image() -> None:
    provider = FakeProvider([[TextDelta(text="I see a pixel")]])
    image = {"name": "screen.png", "mime": "image/png", "data_b64": PIXEL}
    screen = {"name": "Screen", "instructions": "Describe what is on the screen.", "tools": "none"}
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "", command=screen, attachments=[image])
        events = until_finished(ws)

    assert events[-1]["status"] == "ok"
    user = provider.calls[0].messages[-1]
    assert [type(part) for part in user.content] == [ImagePart]


def test_command_block_sits_after_the_summary() -> None:
    text = build_system_prompt(
        None,
        "desktop",
        [],
        now=datetime(2026, 1, 1, tzinfo=UTC),
        summary="We talked about cats.",
        command="Answer briefly.",
    )

    assert text.index("# Earlier in this conversation") < text.index("# Command")
    assert text.index("# Command") < text.index("# Current time")


def test_a_turn_without_tools_is_told_to_answer_in_text() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "hi", command=QUICK)
        until_finished(ws)

    text = system_text(provider.calls[0])
    assert "You have no tools in this turn" in text
    assert "# Expression" not in text
    assert "set_expression" not in text


def test_a_turn_with_tools_keeps_the_expression_rules() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider, small_config()) as (client, state), connected(client, state) as ws:
        send(ws, "hi", command={"name": "Do", "instructions": "Act."})
        until_finished(ws)

    text = system_text(provider.calls[0])
    assert "# Expression" in text
    assert "You have no tools in this turn" not in text


def test_a_character_without_expressions_gets_no_expression_tool() -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    character = {"name": "Phantasm", "persona": "Calm."}
    plain = {**HELLO, "client": {**HELLO["client"], "character": character}}
    with running(provider) as (client, state):
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json(plain)
            assert ws.receive_json()["type"] == "hello.ok"
            send(ws, "hi")
            finished = until_finished(ws)[-1]

    (call,) = provider.calls
    assert "set_expression" not in [tool.name for tool in call.tools]
    assert "# Expression" not in system_text(call)
    assert "Calm." in system_text(call)
    assert finished["status"] == "ok"
    assert finished.get("expression") is None
