"""The scheduler's background runner: starts the turns of schedules that have come due."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from phantasm import clock
from phantasm.store.schedules import Schedule, SchedulesRepo

TICK_S = 30.0
MIN_PAUSE_S = 0.5

logger = logging.getLogger(__name__)

StartTask = Callable[[Schedule], Awaitable[None]]


class SchedulerRunner:
    """Every ``tick_s`` seconds, runs the schedules that are due.

    A run is recorded before its task starts, so a crash in the middle never
    makes the same run happen again, and a schedule missed while Phantasm was off
    runs once rather than once for every missed time.
    """

    def __init__(self, schedules: SchedulesRepo, start: StartTask, tick_s: float = TICK_S) -> None:
        self._schedules = schedules
        self._start = start
        self._tick_s = tick_s
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """True while the background loop is alive."""
        return self._task is not None and not self._task.done()

    async def tick(self) -> list[Schedule]:
        """Start every schedule that is due now; return them."""
        now = clock.now()
        due = await self._schedules.due(now)
        for schedule in due:
            try:
                await self._schedules.mark_run(schedule.id, now)
                await self._start(schedule)
            except Exception:
                logger.exception("Schedule %s could not start", schedule.id)
        return due

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("The scheduler could not check its schedules")
            await asyncio.sleep(await self._pause())

    async def _pause(self) -> float:
        """Seconds until the next check: a tick, or less if a schedule is due sooner."""
        try:
            upcoming = await self._schedules.next_due()
        except Exception:
            return self._tick_s
        if upcoming is None:
            return self._tick_s
        wait = (upcoming - clock.now()).total_seconds()
        return min(self._tick_s, max(wait, MIN_PAUSE_S))

    def start(self) -> None:
        """Begin checking in the background."""
        if not self.running:
            self._task = asyncio.create_task(self._loop(), name="scheduler")

    async def stop(self) -> None:
        """Stop checking; a turn already started keeps running."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
