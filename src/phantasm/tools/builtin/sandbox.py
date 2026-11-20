"""``run_python``: try Python in a sandbox before giving code to the user.

The program runs with Phantasm's own Python (``-I``: no user site, no PYTHON* variables) in
a fresh temporary folder under the macOS sandbox (``sandbox-exec``): no network, no writing
outside that folder, and no reading the user's home folder or other volumes, except the
interpreter itself. It is stopped after ``timeout_s`` seconds of wall time (and as many of
CPU time), and no file it writes may grow past 10 MB. Since nothing it does reaches the
user's data, it needs no approval.
"""

import asyncio
import contextlib
import json
import os
import resource
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, Field

from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
DEFAULT_TIMEOUT_S = 10
MAX_TIMEOUT_S = 60
MAX_FILE_BYTES = 10 * 1024 * 1024
OUTPUT_CHARS = 20_000
POLL_S = 0.05
NO_SANDBOX = "This computer has no sandbox-exec, so the code was not run."


class RunPythonInput(BaseModel):
    code: str = Field(
        min_length=1,
        description="A whole Python program. Put the tests in it: asserts, or prints to compare",
    )
    stdin: str = Field(default="", description="What the program reads from standard input")
    timeout_s: int = Field(
        default=DEFAULT_TIMEOUT_S,
        ge=1,
        le=MAX_TIMEOUT_S,
        description="Stop after this many seconds",
    )


def _quoted(path: Path) -> str:
    """A path as a string literal of the sandbox profile language."""
    return json.dumps(str(path))


def sandbox_profile(folder: Path, home: Path, readable: list[Path]) -> str:
    """The rules: everything is allowed but the network, writing outside ``folder``, and
    reading files or listing folders in ``home`` or other volumes (their paths can still be
    walked through); ``folder`` and ``readable`` stay readable. Only the interpreter in
    ``readable`` may be started, and no Apple events or app launches go out, so the program
    cannot ask another app to do what the sandbox forbids. Later rules win over earlier
    ones."""
    allowed = " ".join(f"(subpath {_quoted(path)})" for path in [folder, *readable])
    runnable = " ".join(f"(subpath {_quoted(path)})" for path in readable)
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny appleevent-send)",
            "(deny lsopen)",
            "(deny process-exec)",
            f"(allow process-exec {runnable})",
            "(deny file-write*)",
            f"(allow file-write* (subpath {_quoted(folder)}))",
            '(allow file-write* (literal "/dev/null") (literal "/dev/dtracehelper"))',
            f'(deny file-read-data (subpath {_quoted(home)}) (subpath "/Volumes"))',
            f"(allow file-read-data {allowed})",
        ]
    )


def _interpreter_folders() -> list[Path]:
    """Where this Python lives: it has to stay readable inside the sandbox."""
    folders = {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.executable).resolve().parent.parent,
    }
    return sorted(folders)


def _limits(cpu_s: int) -> None:
    """Runs in the child before Python starts: CPU time and file size limits."""
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))


class RunPython(Tool[RunPythonInput]):
    """Runs a Python program in a sandbox and reports what it printed."""

    name = "run_python"
    description = (
        "Run a Python program in a sandbox and get what it printed (stdout and stderr "
        "together) and its exit code. Use it to check code before you give it to the user: "
        "put the code and its tests (asserts, sample inputs) in one program, run it, fix what "
        "fails and run again. The sandbox has the standard library only, no network, no "
        "access to the user's files, a fresh empty folder as the working directory, and a "
        "time limit. It needs no approval."
    )
    input_model = RunPythonInput
    default_policy = "allow"

    def label(self, params: RunPythonInput) -> str:
        lines = len(params.code.strip().splitlines())
        return f"python · {lines} line{'s' if lines != 1 else ''}"

    async def run(self, params: RunPythonInput, ctx: ToolContext) -> ToolResult:
        if not Path(SANDBOX_EXEC).exists():
            return error_result(NO_SANDBOX)
        folder = Path(tempfile.mkdtemp(prefix="phantasm-python-")).resolve()
        try:
            (folder / "main.py").write_text(params.code, encoding="utf-8")
            return await self._execute(params, folder, ctx)
        finally:
            await asyncio.to_thread(shutil.rmtree, folder, True)

    async def _execute(self, params: RunPythonInput, folder: Path, ctx: ToolContext) -> ToolResult:
        profile = sandbox_profile(folder, Path.home().resolve(), _interpreter_folders())
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(folder),
            "TMPDIR": str(folder),
            "LANG": "en_US.UTF-8",
        }
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            SANDBOX_EXEC,
            "-p",
            profile,
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "main.py",
            cwd=folder,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: _limits(params.timeout_s + 1),
        )
        talk = asyncio.ensure_future(process.communicate(params.stdin.encode()))
        stopped: str | None = None
        while not talk.done():
            await asyncio.wait({talk}, timeout=POLL_S)
            if talk.done():
                break
            if ctx.cancel.cancelled:
                stopped = "cancelled"
            elif time.monotonic() - started > params.timeout_s:
                stopped = f"stopped after {params.timeout_s} s: too slow, or it never ends"
            if stopped is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                break
        output_bytes, _ = await talk
        elapsed = time.monotonic() - started
        output = output_bytes.decode("utf-8", errors="replace")
        if len(output) > OUTPUT_CHARS:
            output = output[:OUTPUT_CHARS] + f"\n… ({len(output) - OUTPUT_CHARS} more characters)"
        code = process.returncode
        head = f"[{stopped}]" if stopped else f"[exit {code} · {elapsed:.2f} s]"
        result = text_result(f"{head}\n{output}".rstrip(), summary=head.strip("[]"))
        if stopped or code != 0:
            return ToolResult(content=result.content, summary=result.summary, is_error=True)
        return result
