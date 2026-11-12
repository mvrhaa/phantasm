"""Tests for the built-in run_command tool. They start real zsh processes."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.turns import FakeApprover, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.base import CANCELLED, ToolContext, ToolResult
from zero.tools.builtin import builtin_registry, shell
from zero.tools.builtin.shell import RunCommand, RunCommandInput
from zero.tools.permissions import PathGuard


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "STOP_GRACE_S", 0.5)


async def run(
    command: str,
    cancel: CancelToken | None = None,
    approver: FakeApprover | None = None,
    **options: Any,
) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=cancel or CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver if approver is not None else FakeApprover("allow"),
    )
    call = ToolCall(id="r", name="run_command", input={"command": command, **options})
    async with asyncio.timeout(20):
        return await builtin_registry().execute(call, ctx)


def wait_for_file(path: Path) -> None:
    """Wait (blocking, a few seconds at most) until the command has written ``path``."""
    for _ in range(200):
        if path.exists() and path.read_text().strip():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} was not written")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def gone(pid: int) -> bool:
    """Wait a little for a process to disappear."""
    for _ in range(50):
        if not alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


pytestmark = pytest.mark.usefixtures("home")


async def test_output_and_exit_code() -> None:
    result = await run("echo hello")

    assert (text_of(result), result.summary, result.is_error) == (
        "hello\n[exit 0]",
        "exit 0",
        False,
    )


async def test_stdout_and_stderr_are_one_stream() -> None:
    result = await run("echo out; echo err >&2; echo again")

    assert text_of(result) == "out\nerr\nagain\n[exit 0]"


async def test_nonzero_exit_is_an_error() -> None:
    result = await run("echo failing; exit 3")

    assert (text_of(result), result.summary, result.is_error) == (
        "failing\n[exit 3]",
        "exit 3",
        True,
    )


async def test_no_output() -> None:
    assert text_of(await run("true")) == "(no output)\n[exit 0]"


async def test_command_killed_by_a_signal() -> None:
    result = await run("kill -KILL $$")

    assert (result.summary, result.is_error) == ("killed by SIGKILL", True)
    assert text_of(result).endswith("[killed by SIGKILL]")


async def test_runs_in_the_home_folder_by_default(home: Path) -> None:
    assert text_of(await run("pwd")) == f"{home}\n[exit 0]"


async def test_runs_in_cwd(home: Path) -> None:
    (home / "project").mkdir()

    assert text_of(await run("pwd", cwd="~/project")) == f"{home / 'project'}\n[exit 0]"


async def test_zsh_features_work() -> None:
    result = await run("for n in 1 2 3; do print -n $n; done; echo; echo ${(U)${:-zsh}}")

    assert text_of(result) == "123\nZSH\n[exit 0]"


async def test_huge_output_keeps_head_and_tail_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "KEEP_BYTES", 1000)

    result = await run("seq 1 100000")

    text = text_of(result)
    assert text.startswith("1\n2\n3\n")
    assert "bytes of output dropped" in text
    assert text.endswith("99999\n100000\n[exit 0]")


async def test_timeout_stops_the_command() -> None:
    started = asyncio.get_running_loop().time()

    result = await run("echo before; sleep 30", timeout_s=1)

    assert asyncio.get_running_loop().time() - started < 5
    assert text_of(result) == "before\n[stopped after the 1 s timeout]"
    assert (result.summary, result.is_error) == ("timed out", True)


async def test_timeout_stops_background_jobs_too(home: Path) -> None:
    result = await run("sleep 30 & echo $! > bg.pid; sleep 30", timeout_s=1)

    assert result.summary == "timed out"
    assert await gone(int((home / "bg.pid").read_text()))


async def test_command_that_ignores_sigterm_is_killed(home: Path) -> None:
    result = await run("trap '' TERM; echo $$ > shell.pid; sleep 30", timeout_s=1)

    assert result.summary == "timed out"
    assert await gone(int((home / "shell.pid").read_text()))


async def test_cancel_stops_the_command_and_keeps_its_output() -> None:
    cancel = CancelToken()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.5)
        cancel.cancel()

    result, _ = await asyncio.gather(run("echo working; sleep 30", cancel), cancel_soon())

    assert text_of(result) == "working\n[stopped: the user cancelled the turn]"
    assert (result.summary, result.cancelled, result.is_error) == ("cancelled", True, True)


async def test_cancelled_turn_starts_nothing(home: Path) -> None:
    cancel = CancelToken()
    cancel.cancel()

    result = await run("touch started", cancel)

    assert text_of(result) == CANCELLED
    assert not (home / "started").exists()


async def test_background_job_of_a_finished_command_is_stopped_at_once(home: Path) -> None:
    started = asyncio.get_running_loop().time()

    result = await run("sleep 30 & echo $! > bg.pid; echo done")

    assert text_of(result) == "done\n[exit 0]"
    assert asyncio.get_running_loop().time() - started < 3
    assert await gone(int((home / "bg.pid").read_text()))


async def test_background_job_with_its_own_output_is_stopped_too(home: Path) -> None:
    result = await run("sleep 30 > /dev/null 2>&1 & echo $! > bg.pid")

    assert result.summary == "exit 0"
    assert await gone(int((home / "bg.pid").read_text()))


@pytest.mark.parametrize(
    ("cwd", "message"),
    [
        ("~/missing", "cwd ~/missing does not exist."),
        ("relative", "'relative' is a relative path"),
        ("/", "/ is outside the folders tools may use"),
    ],
)
async def test_bad_cwd_is_refused_without_asking(cwd: str, message: str) -> None:
    approver = FakeApprover("allow")

    result = await run("echo hi", approver=approver, cwd=cwd)

    assert result.is_error
    assert text_of(result).startswith(message)
    assert approver.asked == []


async def test_cwd_that_is_a_file_is_refused(home: Path) -> None:
    (home / "file.txt").write_text("x")

    assert text_of(await run("ls", cwd="~/file.txt")) == "cwd ~/file.txt is not a folder."


async def test_command_asks_first_and_denied_ones_do_not_run(home: Path) -> None:
    approver = FakeApprover("deny")

    result = await run("touch ran", approver=approver)

    assert result.denied
    assert not (home / "ran").exists()
    assert [(ask.tool, ask.label) for ask in approver.asked] == [("run_command", "$ touch ran")]


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("ls -la", "$ ls -la"),
        ("  git status\n  git diff", "$ git status …"),
        ("echo " + "x" * 200, "$ echo " + "x" * 72 + "…"),
    ],
)
def test_label(command: str, label: str) -> None:
    shown = RunCommand().label(RunCommandInput(command=command))

    assert shown == label
    assert len(shown) <= shell.LABEL_CHARS


async def test_timeout_is_limited() -> None:
    result = await run("true", timeout_s=shell.MAX_TIMEOUT_S + 1)

    assert "timeout_s: Input should be less than or equal to 3600" in text_of(result)


def test_cancelling_the_turn_kills_the_command(home: Path) -> None:
    call = ToolCall(id="c1", name="run_command", input={"command": "echo $$ > pid; sleep 60"})
    provider = FakeProvider([[ToolCallEvent(call=call)], [TextDelta(text="never")]])
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            ws.send_json({"type": "message.send", "text": "wait", "source": "desktop"})
            events = [ws.receive_json()]
            while events[-1]["type"] != "turn.finished":
                event = events[-1]
                if event["type"] == "approval.request":
                    ws.send_json(
                        {
                            "type": "approval.respond",
                            "approval_id": event["approval_id"],
                            "decision": "allow",
                        }
                    )
                if event["type"] == "tool.started":
                    turn_id = event["turn_id"]
                events.append(ws.receive_json())
                if events[-1]["type"] == "approval.resolved":
                    wait_for_file(home / "pid")
                    ws.send_json({"type": "turn.cancel", "turn_id": turn_id})

    pid = int((home / "pid").read_text())
    assert [e["status"] for e in events if e["type"] == "tool.finished"] == ["cancelled"]
    assert events[-1]["status"] == "cancelled"
    assert len(provider.calls) == 1
    assert not alive(pid)
