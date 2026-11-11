"""Built-in ``run_command``: a shell command in ``zsh -lc``, with a timeout and cancellation.

The command runs in its own process group, so stopping it reaches everything
it started: on timeout or cancel the group gets SIGTERM and, if it is still
alive ``STOP_GRACE_S`` seconds later, SIGKILL. When the command ends by itself,
whatever it left running in the background is stopped the same way, so nothing
it started outlives the call. stdout and stderr are read as
one stream; only the head and the tail of a huge output are kept in memory.
"""

import asyncio
import contextlib
import os
import shutil
import signal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from zero.core.cancellation import CancelToken
from zero.protocol.messages import TextPart
from zero.tools.base import Preview, Tool, ToolContext, ToolResult, error_result
from zero.tools.permissions import PathDenied, PathGuard

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 3600
STOP_GRACE_S = 3.0  # between SIGTERM and SIGKILL
DRAIN_S = 1.0  # how long to wait for the last output once the shell is gone
POLL_S = 0.05  # how often to look whether the shell has exited
KEEP_BYTES = 128 * 1024  # of the head and of the tail of the output
LABEL_CHARS = 80

Outcome = Literal["exit", "timeout", "cancelled"]


def find_zsh() -> str:
    """The zsh to run commands with."""
    return shutil.which("zsh") or "/bin/zsh"


class RunCommandInput(BaseModel):
    command: str = Field(min_length=1, description="Shell command, run with zsh -lc")
    cwd: str = Field(
        default="~", description="Folder to run it in: an absolute path or one that starts with ~"
    )
    timeout_s: int = Field(
        default=DEFAULT_TIMEOUT_S,
        ge=1,
        le=MAX_TIMEOUT_S,
        description="Seconds before the command is stopped",
    )


class _Output:
    """Collects output, keeping only its head and tail once it grows large."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = KEEP_BYTES - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        self.tail += chunk
        if len(self.tail) > KEEP_BYTES:
            del self.tail[: len(self.tail) - KEEP_BYTES]

    def text(self) -> str:
        dropped = self.total - len(self.head) - len(self.tail)
        middle = f"\n[… {dropped} bytes of output dropped …]\n" if dropped else ""
        return self.head.decode(errors="replace") + middle + self.tail.decode(errors="replace")


async def _read(stream: asyncio.StreamReader, output: _Output) -> None:
    while chunk := await stream.read(64 * 1024):
        output.add(chunk)


def _signal_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, sig)


def _group_alive(process: asyncio.subprocess.Process) -> bool:
    """True while any process of the command's group is still running."""
    try:
        os.killpg(process.pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def _exited(process: asyncio.subprocess.Process) -> None:
    """Return once the shell itself has exited.

    ``Process.wait()`` is not used for this: called before the exit, it also
    waits until the output is closed, which a background job can hold open.
    """
    while process.returncode is None:
        await asyncio.sleep(POLL_S)


async def _stop_group(process: asyncio.subprocess.Process, reading: asyncio.Task[None]) -> None:
    """SIGTERM what is left of the group; SIGKILL it if it outlives the grace period.

    The output reaching its end means every process of the group has closed
    it, which is how the end of the whole group is noticed.
    """
    if not _group_alive(process):
        return
    _signal_group(process, signal.SIGTERM)
    ended = asyncio.gather(_exited(process), asyncio.shield(reading))
    try:
        await asyncio.wait_for(ended, STOP_GRACE_S)
    except TimeoutError:
        _signal_group(process, signal.SIGKILL)
        await _exited(process)


async def _execute(command: str, cwd: Path, timeout_s: int, cancel: CancelToken) -> ToolResult:
    cancel.raise_if_cancelled()  # a cancelled turn starts nothing
    process = await asyncio.create_subprocess_exec(
        find_zsh(),
        "-lc",
        command,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    output = _Output()
    reading = asyncio.create_task(_read(process.stdout, output))
    exiting = asyncio.create_task(_exited(process))
    cancelled = asyncio.create_task(cancel.wait())
    try:
        await asyncio.wait(
            {exiting, cancelled}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if exiting.done():
            outcome: Outcome = "exit"
        else:
            outcome = "cancelled" if cancel.cancelled else "timeout"
        # Stops the command itself, or whatever it left running in the background.
        await _stop_group(process, reading)
        await asyncio.wait({reading}, timeout=DRAIN_S)
        return _result(outcome, process.returncode, output, timeout_s)
    finally:
        for task in (reading, exiting, cancelled):
            task.cancel()
        if process.returncode is None:
            _signal_group(process, signal.SIGKILL)
            await _exited(process)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), DRAIN_S)  # let asyncio close the pipes


def _result(outcome: Outcome, code: int | None, output: _Output, timeout_s: int) -> ToolResult:
    text = output.text().rstrip("\n") or "(no output)"
    if outcome == "timeout":
        return _with_status(text, f"[stopped after the {timeout_s} s timeout]", "timed out", True)
    if outcome == "cancelled":
        result = _with_status(text, "[stopped: the user cancelled the turn]", "cancelled", True)
        result.cancelled = True
        return result
    assert code is not None
    if code < 0:
        name = signal.Signals(-code).name if -code in signal.valid_signals() else str(-code)
        return _with_status(text, f"[killed by {name}]", f"killed by {name}", True)
    return _with_status(text, f"[exit {code}]", f"exit {code}", code != 0)


def _with_status(text: str, status: str, summary: str, is_error: bool) -> ToolResult:
    return ToolResult(
        content=[TextPart(text=f"{text}\n{status}")], is_error=is_error, summary=summary
    )


def _check_cwd(raw: str, guard: PathGuard) -> Path | str:
    try:
        cwd = guard.check(raw)
    except PathDenied as exc:
        return str(exc)
    if not cwd.is_dir():
        return f"cwd {raw} is not a folder." if cwd.exists() else f"cwd {raw} does not exist."
    return cwd


class RunCommand(Tool[RunCommandInput]):
    """Runs a shell command and reports its output and exit code."""

    name = "run_command"
    description = (
        "Run a shell command with zsh -lc in cwd (your home folder by default) and get "
        "its output (stdout and stderr together) and exit code. The command is stopped "
        "after timeout_s seconds or when the user cancels, and anything it starts in the "
        "background is stopped when it ends, so do not use it to start servers or daemons. "
        "The user usually approves each command first."
    )
    input_model = RunCommandInput
    default_policy = "ask"

    def label(self, params: RunCommandInput) -> str:
        first, *rest = params.command.strip().splitlines() or [""]
        line = f"$ {first}" + (" …" if rest else "")
        return line if len(line) <= LABEL_CHARS else line[: LABEL_CHARS - 1] + "…"

    async def preview(self, params: RunCommandInput, ctx: ToolContext) -> Preview:
        cwd = await asyncio.to_thread(_check_cwd, params.cwd, ctx.paths)
        return Preview(refusal=cwd) if isinstance(cwd, str) else Preview()

    async def run(self, params: RunCommandInput, ctx: ToolContext) -> ToolResult:
        cwd = await asyncio.to_thread(_check_cwd, params.cwd, ctx.paths)
        if isinstance(cwd, str):
            return error_result(cwd)
        return await _execute(params.command, cwd, params.timeout_s, ctx.cancel)
