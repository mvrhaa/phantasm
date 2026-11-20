"""run_python: Python in the macOS sandbox, without approval."""

import asyncio
import textwrap
from pathlib import Path
from typing import Any

import pytest

from phantasm.config import Config
from phantasm.core.cancellation import CancelToken
from phantasm.providers.base import ToolCall
from phantasm.tools.base import ToolContext, ToolResult
from phantasm.tools.builtin import builtin_registry
from phantasm.tools.builtin.sandbox import SANDBOX_EXEC, sandbox_profile
from phantasm.tools.permissions import PathGuard
from tests.turns import text_of

pytestmark = pytest.mark.skipif(not Path(SANDBOX_EXEC).exists(), reason="macOS only")


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


async def run(code: str, **options: Any) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )
    call = ToolCall(id="p", name="run_python", input={"code": textwrap.dedent(code), **options})
    async with asyncio.timeout(30):
        return await builtin_registry().execute(call, ctx)


def blocked(name: str) -> str:
    return f"""
    def attempt():
        {name}
    try:
        attempt()
        print("allowed")
    except PermissionError:
        print("blocked")
    """


async def test_the_program_runs_with_its_input() -> None:
    result = await run("print(input() * 2)", stdin="ab\n")
    assert not result.is_error
    assert text_of(result).splitlines()[1:] == ["abab"]
    assert result.summary.startswith("exit 0")


async def test_a_failing_test_is_an_error_with_its_traceback() -> None:
    result = await run("assert 1 + 1 == 3, 'math is off'")
    assert result.is_error
    assert text_of(result).startswith("[exit 1")
    assert "AssertionError: math is off" in text_of(result)


async def test_an_endless_program_is_stopped() -> None:
    result = await run("while True:\n    pass", timeout_s=1)
    assert result.is_error
    assert "stopped after 1 s" in text_of(result)


async def test_it_writes_only_in_its_own_folder(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    code = f"""
    open("inside.txt", "w").write("x")
    print("inside ok")
    try:
        open({str(outside)!r}, "w").write("x")
        print("allowed")
    except PermissionError:
        print("blocked")
    """
    result = await run(code)
    assert text_of(result).splitlines()[1:] == ["inside ok", "blocked"]
    assert not outside.exists()


async def test_it_has_no_network() -> None:
    result = await run(
        blocked('import socket; socket.create_connection(("127.0.0.1", 9), timeout=2)')
    )
    assert text_of(result).splitlines()[-1] == "blocked"


async def test_it_cannot_read_the_home_folder(home: Path) -> None:
    (home / "secret.txt").write_text("hidden")
    result = await run(blocked(f"open({str(home / 'secret.txt')!r}).read()"))
    assert text_of(result).splitlines()[-1] == "blocked"


async def test_it_cannot_start_other_programs() -> None:
    result = await run(blocked('import subprocess; subprocess.run(["/bin/ls"], check=True)'))
    assert text_of(result).splitlines()[-1] == "blocked"


def test_it_runs_without_approval_for_every_turn() -> None:
    tool = builtin_registry().get("run_python")
    assert tool is not None
    assert tool.default_policy == "allow"
    assert tool.core


def test_the_profile_closes_the_ways_out() -> None:
    profile = sandbox_profile(Path("/w"), Path("/Users/u"), [Path("/py")])
    for rule in ("(deny network*)", "(deny appleevent-send)", "(deny process-exec)"):
        assert rule in profile
    assert '(allow file-write* (subpath "/w"))' in profile
    assert '(allow process-exec (subpath "/py"))' in profile
