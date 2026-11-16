"""Zero's browser: the user's own Chrome, through Zero's extension (``extension/`` of the project).

The extension connects to ``/browser`` with the token and runs what the brain
asks: open a page, read it, click, type, scroll, take a picture of it. It works
in a Chrome window of its own (behind the user's), so the user's tabs are never
touched, while sites see the user signed in as they are. One extension
connection at a time; a new one replaces the old.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any, Protocol

from zero.core.cancellation import CancelToken

logger = logging.getLogger(__name__)

CONNECT_WAIT_S = 15.0
CALL_TIMEOUT_S = 60.0
NOT_CONNECTED = (
    "Chrome is not connected. Ask the user to open Chrome with Zero's extension: load it "
    "unpacked from the extension folder of Zero's project in chrome://extensions, then put "
    "the token (zero token) into the extension's popup."
)


class BrowserError(Exception):
    """The browser is not there, did not answer, or the page refused the step."""


class Socket(Protocol):
    async def send_text(self, data: str) -> None: ...

    async def receive_text(self) -> str: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class BrowserBridge:
    """The extension's connection and the calls waiting for its answers."""

    def __init__(self) -> None:
        self._socket: Socket | None = None
        self._waiting: dict[str, asyncio.Future[Any]] = {}
        self._connected = asyncio.Event()
        self._next = 0

    @property
    def connected(self) -> bool:
        return self._socket is not None

    async def attach(self, socket: Socket) -> None:
        """Serve one extension connection until it ends."""
        old, self._socket = self._socket, socket
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close(code=1000, reason="replaced")
        self._connected.set()
        logger.info("Chrome connected")
        try:
            while True:
                try:
                    message = json.loads(await socket.receive_text())
                except ValueError:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("type") == "ping":
                    await socket.send_text(json.dumps({"type": "pong"}))
                    continue
                future = self._waiting.get(str(message.get("id")))
                if future is None or future.done():
                    continue
                if message.get("error"):
                    future.set_exception(BrowserError(str(message["error"])))
                else:
                    future.set_result(message.get("result"))
        except Exception:  # the extension went away: its disconnect ends the loop
            pass
        finally:
            if self._socket is socket:
                self._socket = None
                self._connected.clear()
                self._fail("Chrome disconnected before it answered")
                logger.info("Chrome disconnected")

    async def call(
        self,
        command: str,
        args: dict[str, Any],
        cancel: CancelToken | None = None,
        timeout: float = CALL_TIMEOUT_S,
    ) -> Any:
        """Ask the extension to do one thing and wait for its answer."""
        if self._socket is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._connected.wait(), CONNECT_WAIT_S)
        socket = self._socket
        if socket is None:
            raise BrowserError(NOT_CONNECTED)
        self._next += 1
        call_id = str(self._next)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiting[call_id] = future
        try:
            await socket.send_text(json.dumps({"id": call_id, "command": command, "args": args}))
            waiters: set[asyncio.Future[Any]] = {future}
            stop = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
            if stop is not None:
                waiters.add(stop)
            try:
                done, _ = await asyncio.wait(
                    waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                if stop is not None:
                    stop.cancel()
            if future in done:
                return future.result()
            raise BrowserError("stopped" if stop in done else f"Chrome did not answer {command}")
        finally:
            self._waiting.pop(call_id, None)

    def _fail(self, why: str) -> None:
        for future in self._waiting.values():
            if not future.done():
                future.set_exception(BrowserError(why))
