"""Tests for the built-in tools that change files: write_file."""

import os
import stat
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.turns import FakeApprover, text_of
from zero.approvals.manager import Decision
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.base import DENIED, NEEDS_APPROVAL, ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.builtin.edit import Target, diff_summary, make_diff
from zero.tools.permissions import PathGuard


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    return home


async def write(
    path: str | Path,
    content: str,
    approver: FakeApprover | None = None,
) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver if approver is not None else FakeApprover("allow"),
    )
    call = ToolCall(id="w", name="write_file", input={"path": str(path), "content": content})
    return await builtin_registry().execute(call, ctx)


def umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


# --- Diffs -------------------------------------------------------------------


def test_diff_of_a_changed_file() -> None:
    diff = make_diff("~/a.txt", Target(Path("/a.txt"), "one\ntwo\nthree\n"), "one\n2\nthree\n")

    assert diff == "\n".join(
        [
            "--- ~/a.txt",
            "+++ ~/a.txt",
            "@@ -1,3 +1,3 @@",
            " one",
            "-two",
            "+2",
            " three",
        ]
    )
    assert diff_summary(diff) == "+1 -1"


def test_new_file_is_all_additions() -> None:
    diff = make_diff("~/new.txt", Target(Path("/new.txt"), None), "a\nb\n")

    assert diff.splitlines() == ["--- /dev/null", "+++ ~/new.txt", "@@ -0,0 +1,2 @@", "+a", "+b"]
    assert diff_summary(diff) == "+2 -0"


def test_replacing_a_binary_file_says_so() -> None:
    diff = make_diff("~/x.bin", Target(Path("/x.bin"), None, binary=True), "text\n")

    assert diff.splitlines()[:2] == [
        "Binary file ~/x.bin will be replaced with text.",
        "--- ~/x.bin",
    ]
    assert diff_summary(diff) == "+1 -0"


def test_summary_ignores_headers() -> None:
    assert diff_summary("--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n+z") == "+2 -1"


# --- Writing -----------------------------------------------------------------


async def test_new_file_is_created_with_its_folders(home: Path) -> None:
    result = await write("~/notes/2026/plan.md", "# Plan\n\n- rest\n")

    assert (home / "notes" / "2026" / "plan.md").read_text() == "# Plan\n\n- rest\n"
    assert (text_of(result), result.summary) == ("Created ~/notes/2026/plan.md (+3 -0).", "+3 -0")


async def test_new_file_gets_the_usual_permissions(home: Path) -> None:
    await write("~/new.txt", "x")

    assert stat.S_IMODE((home / "new.txt").stat().st_mode) == 0o666 & ~umask()


async def test_existing_file_is_replaced_and_keeps_its_permissions(home: Path) -> None:
    target = home / "config.toml"
    target.write_text("a = 1\nb = 2\n")
    target.chmod(0o640)

    result = await write("~/config.toml", "a = 1\nb = 3\n")

    assert target.read_text() == "a = 1\nb = 3\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert (text_of(result), result.summary) == ("Wrote ~/config.toml (+1 -1).", "+1 -1")
    assert [p.name for p in home.iterdir() if p.name.endswith(".tmp")] == []


async def test_content_is_written_exactly(home: Path) -> None:
    content = "привет\r\nmixed\nends without newline"

    await write("~/exact.txt", content)

    assert (home / "exact.txt").read_bytes() == content.encode("utf-8")


async def test_same_content_writes_nothing(home: Path) -> None:
    target = home / "same.txt"
    target.write_text("hello\n")
    before = target.stat().st_mtime_ns

    result = await write("~/same.txt", "hello\n")

    assert (text_of(result), result.summary) == (
        "~/same.txt already has this content; nothing was written.",
        "no changes",
    )
    assert target.stat().st_mtime_ns == before


async def test_binary_file_can_be_replaced(home: Path) -> None:
    (home / "blob.bin").write_bytes(b"\x00\x01\x02")

    result = await write("~/blob.bin", "now text\n")

    assert (home / "blob.bin").read_text() == "now text\n"
    assert text_of(result) == "Wrote ~/blob.bin (+1 -0)."


async def test_write_through_a_symlink_changes_its_target(home: Path) -> None:
    (home / "real.txt").write_text("old\n")
    (home / "link.txt").symlink_to(home / "real.txt")

    await write("~/link.txt", "new\n")

    assert (home / "real.txt").read_text() == "new\n"
    assert (home / "link.txt").is_symlink()


# --- Approval ----------------------------------------------------------------


async def test_approval_shows_the_diff(home: Path) -> None:
    (home / "todo.txt").write_text("milk\n")
    approver = FakeApprover("allow")

    await write("~/todo.txt", "milk\nbread\n", approver)

    (ask,) = approver.asked
    assert (ask.tool, ask.label, ask.reason) == (
        "write_file",
        "~/todo.txt",
        "write_file is set to ask the user first",
    )
    assert ask.diff == "--- ~/todo.txt\n+++ ~/todo.txt\n@@ -1 +1,2 @@\n milk\n+bread"


async def test_denied_write_changes_nothing(home: Path) -> None:
    result = await write("~/todo.txt", "milk\n", FakeApprover("deny"))

    assert (text_of(result), result.denied) == (DENIED, True)
    assert not (home / "todo.txt").exists()


async def test_without_an_approver_nothing_is_written(home: Path) -> None:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id=None,
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )
    call = ToolCall(id="w", name="write_file", input={"path": "~/x.txt", "content": "x"})

    result = await builtin_registry().execute(call, ctx)

    assert text_of(result) == f"write_file is set to ask the user first, {NEEDS_APPROVAL}"
    assert not (home / "x.txt").exists()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("~/.ssh/authorized_keys", "~/.ssh/authorized_keys is in a protected location."),
        ("notes.md", "'notes.md' is a relative path"),
        ("~", "~ is a folder, not a file."),
    ],
)
async def test_impossible_writes_are_refused_without_asking(
    home: Path, path: str, message: str
) -> None:
    approver = FakeApprover("allow")

    result = await write(path, "x", approver)

    assert result.is_error
    assert text_of(result).startswith(message)
    assert approver.asked == []


async def test_path_outside_the_sandbox_is_refused_without_asking(home: Path) -> None:
    approver = FakeApprover("allow")

    result = await write(home.parent / "outside.txt", "x", approver)

    assert "outside the folders tools may use" in text_of(result)
    assert approver.asked == []
    assert not (home.parent / "outside.txt").exists()


async def test_symlink_pointing_outside_is_refused_without_asking(home: Path) -> None:
    (home / "escape.txt").symlink_to(home.parent / "victim.txt")
    approver = FakeApprover("allow")

    result = await write("~/escape.txt", "pwned", approver)

    assert "outside the folders tools may use" in text_of(result)
    assert approver.asked == []
    assert not (home.parent / "victim.txt").exists()


async def test_read_file_set_to_ask_also_refuses_outside_paths_without_asking(home: Path) -> None:
    config = Config()
    config.permissions.tool_policies["read_file"] = "ask"
    approver = FakeApprover("allow")
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver,
    )
    call = ToolCall(id="r", name="read_file", input={"path": "/etc/hosts"})

    result = await builtin_registry().execute(call, ctx)

    assert "outside the folders tools may use" in text_of(result)
    assert approver.asked == []


# --- Over the WebSocket ------------------------------------------------------


def write_script(content: str) -> list[list[Any]]:
    call = ToolCall(id="w1", name="write_file", input={"path": "~/hello.txt", "content": content})
    return [[ToolCallEvent(call=call)], [TextDelta(text="Saved.")]]


@pytest.mark.parametrize(("decision", "written"), [("allow", True), ("deny", False)])
def test_write_file_over_websocket(home: Path, decision: Decision, written: bool) -> None:
    provider = FakeProvider(write_script("hi there\n"))
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            ws.send_json({"type": "message.send", "text": "save hello", "source": "desktop"})
            events = [ws.receive_json()]
            while events[-1]["type"] != "turn.finished":
                if events[-1]["type"] == "approval.request":
                    request = events[-1]
                    ws.send_json(
                        {
                            "type": "approval.respond",
                            "approval_id": request["approval_id"],
                            "decision": decision,
                        }
                    )
                events.append(ws.receive_json())

    assert request["diff"] == "--- /dev/null\n+++ ~/hello.txt\n@@ -0,0 +1 @@\n+hi there"
    finished = [e for e in events if e["type"] == "tool.finished"]
    assert [e["status"] for e in finished] == ["ok" if written else "denied"]
    assert (home / "hello.txt").exists() is written
