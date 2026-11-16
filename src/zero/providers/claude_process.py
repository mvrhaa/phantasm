"""One ``claude -p`` process that speaks stream-json on its stdin and stdout.

Claude Code asks things of its host while it works (control requests: may a
tool run, an MCP message for the host's own server). They are answered at
once by ``on_control``; everything else it prints waits in ``events``, which
ends with ``None`` once the process is gone.
"""

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LINE_LIMIT = 64 * 1024 * 1024
STDERR_LINES = 20
CLOSE_WAIT_S = 10.0

OnControl = Callable[["ClaudeProcess", dict[str, Any]], Awaitable[None]]


def child_env(extra: dict[str, str]) -> dict[str, str]:
    """The brain's environment for Claude Code, without the marks of a Claude Code
    that may have started the brain (they would change how the child behaves)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "CLAUDECODE" and not key.startswith("CLAUDE_CODE_")
    }
    env.update(extra)
    return env


class ClaudeProcess:
    """A running Claude Code; build it with ``start``."""

    def __init__(self, process: asyncio.subprocess.Process, on_control: OnControl) -> None:
        self._process = process
        self._on_control = on_control
        self._write = asyncio.Lock()
        self.events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.errors: deque[str] = deque(maxlen=STDERR_LINES)
        self._tasks = [
            asyncio.create_task(self._read(), name="claude-code-out"),
            asyncio.create_task(self._read_errors(), name="claude-code-err"),
        ]

    @classmethod
    async def start(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        on_control: OnControl,
    ) -> "ClaudeProcess":
        """Start Claude Code and send it the ``initialize`` request."""
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=LINE_LIMIT,
            start_new_session=True,
        )
        claude = cls(process, on_control)
        await claude.send(
            {
                "type": "control_request",
                "request_id": "init",
                "request": {"subtype": "initialize", "hooks": None},
            }
        )
        return claude

    @property
    def alive(self) -> bool:
        return self._process.returncode is None

    async def _read(self) -> None:
        stdout = self._process.stdout
        assert stdout is not None
        try:
            while line := await stdout.readline():
                try:
                    event = json.loads(line)
                except ValueError:
                    self.errors.append(line.decode("utf-8", "replace").rstrip())
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "control_request":
                    try:
                        await self._on_control(self, event)
                    except Exception:
                        logger.exception("Answering a request of Claude Code")
                    continue
                await self.events.put(event)
        except (ValueError, OSError):
            logger.warning("Claude Code printed a line that could not be read")
        finally:
            with contextlib.suppress(ProcessLookupError):
                await self._process.wait()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._tasks[1]), timeout=2)
            await self.events.put(None)

    async def _read_errors(self) -> None:
        stderr = self._process.stderr
        assert stderr is not None
        with contextlib.suppress(ValueError, OSError):
            while line := await stderr.readline():
                self.errors.append(line.decode("utf-8", "replace").rstrip())

    async def send(self, message: dict[str, Any]) -> None:
        """Write one message; a process that is gone shows it in ``events``, not here."""
        stdin = self._process.stdin
        if stdin is None or stdin.is_closing():
            return
        async with self._write:
            try:
                stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
                await stdin.drain()
            except (OSError, RuntimeError):
                pass

    async def say(self, content: list[dict[str, Any]]) -> None:
        """A user message: text and image blocks."""
        await self.send(
            {
                "type": "user",
                "session_id": "",
                "parent_tool_use_id": None,
                "message": {"role": "user", "content": content},
            }
        )

    async def answer(
        self, request_id: Any, response: dict[str, Any] | None = None, error: str = ""
    ) -> None:
        """Answer a control request."""
        body: dict[str, Any] = (
            {"subtype": "error", "request_id": request_id, "error": error}
            if error
            else {"subtype": "success", "request_id": request_id, "response": response or {}}
        )
        await self.send({"type": "control_response", "response": body})

    async def close(self) -> None:
        """Let it end on its own (it saves its session), or kill it after a while."""
        stdin = self._process.stdin
        if stdin is not None and not stdin.is_closing():
            stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=CLOSE_WAIT_S)
        except TimeoutError:
            self.kill()
            with contextlib.suppress(ProcessLookupError):
                await self._process.wait()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def kill(self) -> None:
        """Stop it now, with anything it started."""
        if not self.alive:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._process.pid, 9)
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()
