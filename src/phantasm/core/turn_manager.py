"""Running turns in the background and keeping track of the ones in progress."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from phantasm.core.turn import Turn

AfterTurn = Callable[[], Awaitable[None]]
"""Follow-up work started once a turn has finished, such as titling its session."""

logger = logging.getLogger(__name__)


class OpenTurns(Protocol):
    """Where turns in progress are noted, so a crash does not lose them."""

    async def add(
        self, turn_id: str, session_id: str, source: str, agent_id: str | None
    ) -> None: ...

    async def remove(self, turn_id: str) -> None: ...


class TurnManager:
    """Runs turns as background tasks and keeps track of the ones in progress.

    With ``open_turns`` each turn is noted there while it runs; a turn that
    finishes in any way, a cancelled one too, is crossed out.
    """

    def __init__(self, open_turns: OpenTurns | None = None) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._turns: dict[str, Turn] = {}
        self._open = open_turns

    @property
    def active(self) -> list[Turn]:
        """Turns that are still running."""
        return list(self._turns.values())

    def get(self, turn_id: str) -> Turn | None:
        """The running turn with this id, or None."""
        return self._turns.get(turn_id)

    def busy(self, session_id: str) -> bool:
        """True while a turn of this session is unfinished.

        A turn counts as finished once it has saved everything, just before it
        announces ``turn.finished``, so a client may send the next message as
        soon as it sees that event.
        """
        return any(t.session_id == session_id and not t.done for t in self._turns.values())

    def start(self, turn: Turn, after: AfterTurn | None = None) -> None:
        """Start a turn in the background and forget it once it is done.

        ``after`` runs in the same task once ``turn.finished`` has gone out, so
        clients get the answer without waiting for it. Its failures are logged.
        """
        task = asyncio.create_task(self._run(turn, after), name=f"turn-{turn.id}")
        self._tasks[turn.id] = task
        self._turns[turn.id] = turn
        task.add_done_callback(lambda _: self._forget(turn.id))

    async def _run(self, turn: Turn, after: AfterTurn | None) -> None:
        if self._open is not None:
            try:
                agent = turn.agent.id if turn.agent is not None else None
                await self._open.add(turn.id, turn.session_id, turn.source, agent)
            except Exception:
                logger.exception("Turn %s could not be noted as open", turn.id)
        try:
            await turn.run()
        finally:
            if self._open is not None:
                try:
                    await self._open.remove(turn.id)
                except Exception:
                    logger.exception("Turn %s could not be crossed out", turn.id)
        if after is None:
            return
        try:
            await after()
        except Exception:
            logger.exception("Follow-up of turn %s failed", turn.id)

    def _forget(self, turn_id: str) -> None:
        self._tasks.pop(turn_id, None)
        self._turns.pop(turn_id, None)

    async def wait_idle(self) -> None:
        """Wait until every running turn has finished."""
        while self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def wait_free(self, session_id: str) -> None:
        """Wait until no unfinished turn runs in this session."""
        while self.busy(session_id):
            running = [
                self._tasks[t.id] for t in self._turns.values() if t.session_id == session_id
            ]
            await asyncio.wait(running)

    async def stop_session(self, session_id: str) -> None:
        """Cancel the turns of a session and wait until they and their follow-ups are over."""
        turns = [turn for turn in self._turns.values() if turn.session_id == session_id]
        for turn in turns:
            turn.cancel.cancel()
        if turns:
            await asyncio.wait([self._tasks[turn.id] for turn in turns])

    async def cancel_all(self, grace_s: float) -> None:
        """Cancel every turn and give them ``grace_s`` seconds to finish properly.

        A cancelled turn saves what it has and announces ``turn.finished``
        with status ``cancelled``; waits for approvals and client actions end
        with it. Turns still running after the grace period are cut off.
        """
        for turn in self.active:
            turn.cancel.cancel()
        tasks = list(self._tasks.values())
        if not tasks:
            return
        _, late = await asyncio.wait(tasks, timeout=grace_s)
        for task in late:
            task.cancel()
        await asyncio.gather(*late, return_exceptions=True)

    async def shutdown(self) -> None:
        """Stop all running turns; used when the application shuts down."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
