"""Choosing who thinks in the user's turns: ``provider.set``, ``hello.ok`` and the turns."""

from pathlib import Path

from phantasm.config import Config, load_config
from phantasm.providers.base import TextDelta
from phantasm.providers.fake import FakeProvider
from phantasm.server.thinker import find_claude, thinker_info
from tests.test_turn_text import HELLO, connected, running, send, until_finished


def answer(events: list[dict]) -> str:
    return "".join(event["delta"] for event in events if event["type"] == "text.delta")


def test_switching_sends_the_turns_to_claude_code_and_back() -> None:
    main = FakeProvider([[TextDelta(text="from main")]] * 4)
    with running(main) as (client, state), connected(client, state) as ws:
        state.claude = FakeProvider([[TextDelta(text="from claude")]])
        ws.send_json({"type": "provider.set", "id": "p1", "claude_code": True})
        on = ws.receive_json()
        send(ws, "hi")
        to_claude = until_finished(ws)
        session_id = to_claude[-1]["session_id"]
        ws.receive_json()  # the session's title, written by the main provider
        ws.send_json({"type": "provider.set", "id": "p2", "claude_code": False})
        off = ws.receive_json()
        send(ws, "again", session_id=session_id)
        to_main = until_finished(ws)
        saved = load_config()

    assert (on["type"], on["reply_to"]) == ("provider.changed", "p1")
    assert on["thinker"] == {"claude_code": True, "name": "Claude Sonnet"}
    assert answer(to_claude) == "from claude"
    assert off["thinker"] == {"claude_code": False, "name": "Fake"}
    assert answer(to_main) == "from main"
    assert saved.claude_code.active is False


def test_choice_is_saved_and_told_in_hello() -> None:
    with running() as (client, state), connected(client, state) as ws:
        state.claude = FakeProvider([])
        ws.send_json({"type": "provider.set", "id": "p1", "claude_code": True})
        ws.receive_json()
    assert load_config().claude_code.active is True
    config = load_config()
    with running(config=config) as (client, state):
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json(HELLO)
            hello = ws.receive_json()
    assert hello["thinker"]["claude_code"] is True


def test_missing_claude_code_is_refused(tmp_path: Path) -> None:
    config = Config()
    config.claude_code.command = str(tmp_path / "no-such-claude")
    with running(config=config) as (client, state), connected(client, state) as ws:
        state.claude = None
        ws.send_json({"type": "provider.set", "id": "p1", "claude_code": True})
        refused = ws.receive_json()
        active = state.config.claude_code.active
    assert (refused["type"], refused["code"], refused["reply_to"]) == (
        "error",
        "provider_not_configured",
        "p1",
    )
    assert active is False


def test_find_claude_looks_in_the_install_folders(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    command = folder / "claude"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr("phantasm.server.thinker.INSTALL_FOLDERS", (str(folder),))
    assert find_claude("claude") == str(command)
    assert find_claude(str(command)) == str(command)
    assert find_claude(str(tmp_path / "missing")) is None


def test_thinker_names() -> None:
    config = Config()
    config.provider.name = "google"
    assert thinker_info(config).name == "Gemini"
    config.claude_code.active = True
    config.claude_code.model = "opus"
    assert thinker_info(config).name == "Claude Opus"
    config.claude_code.model = "claude-sonnet-5-5"
    assert thinker_info(config).name == "claude-sonnet-5-5"
