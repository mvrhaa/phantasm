"""Tests for background jobs: run_command going on in the background, job_output, job_stop.

They start real zsh processes; a command goes to the background after half a second.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from tests.turns import FakeApprover, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.providers.base import ToolCall
from zero.tools.base import ToolContext, ToolResult
from zero.tools.builtin import builtin_registry, jobs, shell
from zero.tools.builtin.jobs import JobManager
from zero.tools.permissions import PathGuard


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def quick_background(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "STOP_GRACE_S", 0.5)
    monkeypatch.setattr(shell, "BACKGROUND_AFTER_S", 0.5)


pytestmark = pytest.mark.usefixtures("home")


class Rig:
    def __init__(self) -> None:
        self.jobs = JobManager()
        self.registry = builtin_registry(jobs=self.jobs)

    async def call(
        self, tool: str, cancel: CancelToken | None = None, **tool_input: Any
    ) -> ToolResult:
        config = Config()
        ctx = ToolContext(
            session_id="s1",
            turn_id="t1",
            client_id="c1",
            cancel=cancel or CancelToken(),
            config=config,
            paths=PathGuard.from_config(config.permissions),
            approvals=FakeApprover("allow"),
        )
        async with asyncio.timeout(20):
            return await self.registry.execute(ToolCall(id="r", name=tool, input=tool_input), ctx)


@pytest.fixture
async def rig() -> Any:
    rig = Rig()
    yield rig
    await rig.jobs.stop_all()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def gone(pid: int) -> bool:
    for _ in range(60):
        if not alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def test_a_long_command_goes_on_as_a_job(rig: Rig) -> None:
    started = await rig.call("run_command", command="echo first; sleep 1; echo second")

    assert not started.is_error
    assert "first" in text_of(started)
    assert "job-1" in text_of(started)
    assert started.summary == "job-1 running"

    done = await rig.call("job_output", job_id="job-1", wait_s=10)
    assert text_of(done) == "second\n[job-1: exit 0]"
    assert (done.summary, done.is_error) == ("exit 0", False)

    again = await rig.call("job_output", job_id="job-1")
    assert text_of(again) == "(no new output)\n[job-1: exit 0]"


async def test_a_quick_command_is_no_job(rig: Rig) -> None:
    result = await rig.call("run_command", command="echo hi")
    assert text_of(result) == "hi\n[exit 0]"
    assert rig.jobs.running() == 0


async def test_job_stop_stops_the_whole_group(rig: Rig, home: Path) -> None:
    await rig.call("run_command", command="sleep 30 & echo $! > bg.pid; sleep 30")
    pid = int((home / "bg.pid").read_text())

    stopped = await rig.call("job_stop", job_id="job-1")

    assert text_of(stopped).endswith("[job-1: stopped]")
    assert stopped.summary == "stopped"
    assert await gone(pid)


async def test_the_timeout_still_ends_a_job(rig: Rig) -> None:
    await rig.call("run_command", command="echo go; sleep 30", timeout_s=2)

    result = await rig.call("job_output", job_id="job-1", wait_s=10)

    assert text_of(result) == "(no new output)\n[job-1: stopped after the 2 s timeout]"
    assert (result.summary, result.is_error) == ("timed out", True)


async def test_failed_job_is_an_error(rig: Rig) -> None:
    await rig.call("run_command", command="sleep 1; echo bad; exit 3")
    result = await rig.call("job_output", job_id="job-1", wait_s=10)
    assert text_of(result) == "bad\n[job-1: exit 3]"
    assert result.is_error


async def test_cancelling_the_turn_leaves_a_job_running(rig: Rig, home: Path) -> None:
    cancel = CancelToken()
    await rig.call("run_command", cancel, command="echo $$ > shell.pid; sleep 30")
    cancel.cancel()
    await asyncio.sleep(0.3)

    running = await rig.call("job_output", job_id="job-1")

    assert text_of(running).startswith("(no new output)\n[job-1: running for ")
    assert alive(int((home / "shell.pid").read_text()))


async def test_without_room_a_command_waits_in_the_foreground(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "MAX_RUNNING_JOBS", 0)
    result = await rig.call("run_command", command="echo hi; sleep 30", timeout_s=1)
    assert result.summary == "timed out"


async def test_unknown_job_names_the_known_ones(rig: Rig) -> None:
    await rig.call("run_command", command="sleep 30")
    result = await rig.call("job_output", job_id="job-9")
    assert result.is_error
    assert "job-1 ($ sleep 30, running" in text_of(result)


async def test_stop_all_ends_every_job(rig: Rig, home: Path) -> None:
    await rig.call("run_command", command="echo $$ > a.pid; sleep 30")
    await rig.call("run_command", command="echo $$ > b.pid; sleep 30")
    pids = [int((home / name).read_text()) for name in ("a.pid", "b.pid")]

    await rig.jobs.stop_all()

    assert rig.jobs.running() == 0
    assert [await gone(pid) for pid in pids] == [True, True]


async def test_a_jobs_whole_output_is_in_its_log(rig: Rig) -> None:
    await rig.call("run_command", command="echo one; sleep 1; echo two")
    await rig.call("job_output", job_id="job-1", wait_s=10)
    log = rig.jobs.get("job-1").log_path  # type: ignore[union-attr]
    assert log.read_text() == "one\ntwo\n"


async def test_the_app_has_job_tools_and_stopping_it_ends_the_jobs(rig: Rig, home: Path) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from zero.server.app import create_app, graceful_stop
    from zero.server.state import get_state

    app = create_app(Config())
    with TestClient(app):
        names = {spec.name for spec in get_state(app).tools.specs()}
    assert {"job_output", "job_stop"} <= names

    await rig.call("run_command", command="echo $$ > job.pid; sleep 30")

    async def nothing(*args: Any) -> None:
        return None

    state = SimpleNamespace(
        stopping=False,
        turns=SimpleNamespace(active=[], cancel_all=nothing),
        scheduler=SimpleNamespace(stop=nothing),
        agent_runner=SimpleNamespace(close=nothing),
        connections=SimpleNamespace(connections=[]),
        jobs=rig.jobs,
    )
    await graceful_stop(state)  # type: ignore[arg-type]

    assert rig.jobs.running() == 0
    assert await gone(int((home / "job.pid").read_text()))
