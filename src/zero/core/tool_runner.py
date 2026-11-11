"""Running one tool call of a turn: what clients see and what is recorded about it."""

import logging
from datetime import datetime
from typing import Literal, Protocol

from zero import clock
from zero.core.events import EventBus
from zero.core.expression import EXPRESSION_SET, EXPRESSION_TOOL, ExpressionTool
from zero.protocol.messages import (
    Expression,
    FilePart,
    FileSent,
    ToolFinished,
    ToolRun,
    ToolStarted,
)
from zero.providers.base import ToolCall
from zero.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    error_result,
    output_preview,
    text_result,
)

ToolStatus = Literal["ok", "error", "denied", "cancelled"]

logger = logging.getLogger(__name__)


class ToolRunRecorder(Protocol):
    """Where finished tool calls are kept for the history."""

    async def add(self, session_id: str, run: ToolRun) -> None: ...


def tool_status(result: ToolResult) -> ToolStatus:
    """The status ``tool.finished`` reports for a result."""
    if result.cancelled:
        return "cancelled"
    if result.denied:
        return "denied"
    return "error" if result.is_error else "ok"


def _duration_ms(started: datetime) -> int:
    return max(0, round((clock.now() - started).total_seconds() * 1000))


class ToolRunner:
    """Runs the tool calls of one turn.

    For each call clients get ``tool.started``, a ``file`` event for every file
    part of the result, and ``tool.finished`` with the summary, a short preview
    of the output and the duration. Visible calls are recorded in ``runs``;
    ``set_expression`` is handled here, is marked hidden and is not recorded.
    """

    def __init__(
        self,
        *,
        session_id: str,
        turn_id: str,
        events: EventBus,
        tools: ToolRegistry,
        context: ToolContext,
        expression: ExpressionTool | None,
        runs: ToolRunRecorder | None,
    ) -> None:
        self._session_id = session_id
        self._turn_id = turn_id
        self._events = events
        self._tools = tools
        self._context = context
        self._expression = expression
        self._runs = runs

    async def run(self, call: ToolCall) -> ToolResult:
        """Run one call and report it; failures come back as error results."""
        expression = self._expression if call.name == EXPRESSION_TOOL else None
        hidden = expression is not None
        label = call.name if hidden else self._tools.label(call)
        started = clock.now()
        await self._events.publish(
            ToolStarted(
                session_id=self._session_id,
                turn_id=self._turn_id,
                call_id=call.id,
                tool=call.name,
                label=label,
                input=call.input,
                hidden=hidden,
            )
        )
        if expression is not None:
            result = await self._set_expression(expression, call)
        else:
            result = await self._tools.execute(call, self._context)

        for part in result.content:
            if isinstance(part, FilePart):
                await self._events.publish(
                    FileSent(
                        session_id=self._session_id,
                        turn_id=self._turn_id,
                        call_id=call.id,
                        file=part,
                    )
                )
        run = ToolRun(
            call_id=call.id,
            turn_id=self._turn_id,
            tool=call.name,
            label=label,
            input=call.input,
            status=tool_status(result),
            summary=result.summary,
            output_preview=output_preview(result),
            duration_ms=_duration_ms(started),
            created_at=started,
        )
        if not hidden:
            await self._record(run)
        await self._events.publish(
            ToolFinished(
                session_id=self._session_id,
                turn_id=self._turn_id,
                call_id=call.id,
                status=run.status,
                summary=run.summary,
                output_preview=run.output_preview,
                duration_ms=run.duration_ms,
            )
        )
        return result

    async def _record(self, run: ToolRun) -> None:
        """Keep the run for the history; a failure here does not break the turn."""
        if self._runs is None:
            return
        try:
            await self._runs.add(self._session_id, run)
        except Exception:
            logger.exception("Could not record tool run %s of turn %s", run.call_id, run.turn_id)

    async def _set_expression(self, expression: ExpressionTool, call: ToolCall) -> ToolResult:
        """Change the character's expression and tell the clients at once."""
        try:
            state = expression.apply(call.input)
        except ValueError as exc:
            return error_result(str(exc))
        await self._events.publish(
            Expression(
                session_id=self._session_id,
                turn_id=self._turn_id,
                pose=state.pose,
                mood=state.mood,
            )
        )
        return text_result(EXPRESSION_SET)
