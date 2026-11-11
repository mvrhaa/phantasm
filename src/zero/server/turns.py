"""Building turns from the application state, for messages and for schedules."""

import logging
from collections.abc import Sequence

from zero.core.events import ReplyTo
from zero.core.turn import Turn
from zero.protocol.messages import ClientInfo, FilePart, SessionCreated, SessionInfo
from zero.providers.base import LLMProvider
from zero.server.state import AppState
from zero.store.messages import Source
from zero.store.schedules import Schedule

TITLE_CHARS = 60

logger = logging.getLogger(__name__)


def make_turn(
    state: AppState,
    provider: LLMProvider,
    *,
    session_id: str,
    text: str,
    source: Source,
    client: ClientInfo | None = None,
    reply: ReplyTo | None = None,
    attachments: Sequence[FilePart] = (),
) -> Turn:
    """A turn wired to everything the application has: tools, approvals, memory and so on."""
    return Turn(
        session_id=session_id,
        text=text,
        source=source,
        client=client,
        config=state.config,
        provider=provider,
        sessions=state.sessions,
        messages=state.messages,
        events=state.events,
        tools=state.tools,
        approvals=state.approvals,
        approval_rules=state.approval_rules,
        tool_runs=state.tool_runs,
        action_caller=state.actions,
        memory=state.memory,
        summaries=state.summaries,
        attachments=attachments,
        reply=reply,
    )


async def start_scheduled_turn(state: AppState, schedule: Schedule) -> None:
    """Run a schedule's task as a turn in the schedule's own session.

    The session ``scheduled-<id>`` is created the first time and announced to
    every client. A run is skipped while the previous one is still going.
    """
    if state.provider is None:
        logger.warning("Schedule %s skipped: no usable provider", schedule.id)
        return
    session_id = f"scheduled-{schedule.id}"
    session = await state.sessions.get(session_id)
    if session is None:
        task = schedule.task
        title = task if len(task) <= TITLE_CHARS else task[: TITLE_CHARS - 1] + "…"
        session = await state.sessions.create(f"Scheduled: {title}", session_id=session_id)
        info = SessionInfo(id=session.id, title=session.title, updated_at=session.updated_at)
        await state.events.publish(SessionCreated(session=info))
    if state.turns.busy(session.id):
        logger.warning("Schedule %s skipped: its previous turn is still running", schedule.id)
        return
    state.turns.start(
        make_turn(
            state, state.provider, session_id=session.id, text=schedule.task, source="scheduler"
        )
    )
