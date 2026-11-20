"""A small MCP client for servers that speak JSON-RPC over stdio (newline-delimited).

It does what Phantasm needs of a server: ``initialize``, ``tools/list`` and
``tools/call``. Requests the server sends back get a short answer (``ping``,
an empty ``roots/list``) or "not supported"; notifications are noted
(``tools/list_changed`` makes the next ``tools`` read the list again).
"""

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
LINE_LIMIT = 32 * 1024 * 1024
START_TIMEOUT_S = 30.0
STDERR_LINES = 20


class McpError(Exception):
    """The server could not be reached, failed or answered with an error."""


class McpClient:
    """One running MCP server; ``start`` it, then ``list_tools`` and ``call``."""

    def __init__(
        self, name: str, command: str, args: Sequence[str] = (), env: dict[str, str] | None = None
    ) -> None:
        self.name = name
        self._argv = [command, *args]
        self._env = env or {}
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._waiting: dict[int, asyncio.Future[Any]] = {}
        self._write = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = []
        self.errors: deque[str] = deque(maxlen=STDERR_LINES)
        self.tools_changed = False

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        """Start the server and do the MCP handshake."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._argv,
                env={**os.environ, **self._env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=LINE_LIMIT,
                start_new_session=True,
            )
        except OSError as exc:
            raise McpError(f"{self.name}: cannot start {self._argv[0]} ({exc.strerror})") from None
        self._tasks = [
            asyncio.create_task(self._read(), name=f"mcp-{self.name}-out"),
            asyncio.create_task(self._read_errors(), name=f"mcp-{self.name}-err"),
        ]
        params = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "phantasm", "version": "1"},
        }
        await self.request("initialize", params, timeout=START_TIMEOUT_S)
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def list_tools(self) -> list[dict[str, Any]]:
        """Every tool of the server, all pages of the list."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += [tool for tool in result.get("tools") or [] if isinstance(tool, dict)]
            cursor = result.get("nextCursor")
            if not cursor:
                self.tools_changed = False
                return tools

    async def call(self, tool: str, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Call a tool; its result (``content``, ``isError``)."""
        return await self.request("tools/call", {"name": tool, "arguments": arguments}, timeout)

    async def request(
        self, method: str, params: dict[str, Any], timeout: float = START_TIMEOUT_S
    ) -> dict[str, Any]:
        if not self.alive:
            raise McpError(f"{self.name} is not running")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            raise McpError(f"{self.name} did not answer {method} in {timeout:.0f} s") from None
        finally:
            self._waiting.pop(request_id, None)

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, 9)
                with contextlib.suppress(ProcessLookupError):
                    await process.wait()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._fail_waiting("the server was closed")

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.stdin.is_closing():
            raise McpError(f"{self.name} is not running")
        async with self._write:
            try:
                process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
                await process.stdin.drain()
            except (OSError, RuntimeError) as exc:
                raise McpError(f"{self.name} stopped ({type(exc).__name__})") from None

    async def _read(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while line := await process.stdout.readline():
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    await self._handle(message)
        except (ValueError, OSError):
            logger.warning("MCP server %s printed a line that could not be read", self.name)
        finally:
            detail = self.errors[-1] if self.errors else "no message"
            self._fail_waiting(f"stopped: {detail}")

    async def _handle(self, message: dict[str, Any]) -> None:
        if "method" not in message:
            future = self._waiting.get(message.get("id"))  # type: ignore[arg-type]
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"] or {}
                future.set_exception(McpError(f"{self.name}: {error.get('message', 'error')}"))
            else:
                future.set_result(message.get("result") or {})
            return
        method = message["method"]
        if "id" not in message:
            if method == "notifications/tools/list_changed":
                self.tools_changed = True
            return
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "ping":
            reply["result"] = {}
        elif method == "roots/list":
            reply["result"] = {"roots": []}
        else:
            reply["error"] = {"code": -32601, "message": f"Phantasm does not support {method}"}
        with contextlib.suppress(McpError):
            await self._send(reply)

    async def _read_errors(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        with contextlib.suppress(ValueError, OSError):
            while line := await process.stderr.readline():
                self.errors.append(line.decode("utf-8", "replace").rstrip())

    def _fail_waiting(self, why: str) -> None:
        for future in self._waiting.values():
            if not future.done():
                future.set_exception(McpError(f"{self.name} {why}"))
