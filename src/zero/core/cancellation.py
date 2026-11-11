"""Cooperative cancellation of a turn."""

import asyncio


class TurnCancelled(Exception):
    """Raised inside a turn when it notices that it was cancelled."""


class CancelToken:
    """A flag a turn passes to everything it runs.

    Code that takes a while checks ``cancelled`` or calls
    ``raise_if_cancelled()`` between steps; code that waits for something
    else can race that wait against ``wait()``. Once set, the token stays set.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        """True once ``cancel()`` has been called."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation. Calling it again changes nothing."""
        self._event.set()

    async def wait(self) -> None:
        """Return when the token is cancelled; at once if it already is."""
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise ``TurnCancelled`` if cancellation was requested."""
        if self.cancelled:
            raise TurnCancelled()
