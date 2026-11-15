"""Building turns from the application state, for messages, schedules and agents."""

import logging
from collections.abc import Sequence

from zero.agents.runner import AgentApprovals
from zero.approvals.manager import Approver
from zero.core.events import ReplyTo
from zero.core.turn import Turn
from zero.protocol.messages import AgentRef, ClientInfo, FilePart, MessageCommand, SessionCreated
from zero.providers.base import LLMProvider
from zero.server.state import AppState
from zero.store.agents import Agent
from zero.store.messages import Source
from zero.store.schedules import Schedule
from zero.tools.base import ToolRegistry
from zero.tools.builtin.agents import AGENT_TOOL_PREFIX, agent_side_tools

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
    agent: AgentRef | None = None,
    tools: ToolRegistry | None = None,
    approvals: Approver | None = None,
    command: MessageCommand | None = None,
) -> Turn:
    """A turn wired to everything the application has: tools, approvals, memory and so on.

    ``tools`` and ``approvals`` replace the shared ones, ``agent`` marks an agent's turn,
    ``command`` came with the user's message.
    """
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
        tools=tools if tools is not None else state.tools,
        approvals=approvals if approvals is not None else state.approvals,
        approval_rules=state.approval_rules,
        tool_runs=state.tool_runs,
        action_caller=state.actions,
        memory=state.memory,
        session_memory=state.session_memory,
        summaries=state.summaries,
        usage=state.usage,
        agent=agent,
        attachments=attachments,
        reply=reply,
        command=command,
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
        session = await state.sessions.create(
            f"Scheduled: {title}", session_id=session_id, kind="scheduled"
        )
        await state.events.publish(SessionCreated(session=session.info()))
    if state.turns.busy(session.id):
        logger.warning("Schedule %s skipped: its previous turn is still running", schedule.id)
        return
    state.turns.start(
        make_turn(
            state, state.provider, session_id=session.id, text=schedule.task, source="scheduler"
        )
    )


async def start_agent_turn(state: AppState, agent: Agent, text: str) -> None:
    """Run ``text`` as a turn in the agent's session.

    The agent has every tool but the ``agent_*`` ones, plus ``ask_parent``; while
    one of its tools waits for the user's approval, its status is ``waiting_user``.
    The agent runner hears how the turn ended once ``turn.finished`` is out.
    """
    if state.provider is None or agent.session_id is None:
        logger.warning("Agent %s cannot work: no provider or no session", agent.id)
        await state.agent_runner.finished(agent.id, None)
        return
    turn = make_turn(
        state,
        state.provider,
        session_id=agent.session_id,
        text=text,
        source="agent",
        agent=AgentRef(id=agent.id, name=agent.name),
        tools=state.tools.without(AGENT_TOOL_PREFIX).extended(agent_side_tools(state.agent_runner)),
        approvals=AgentApprovals(state.approvals, state.agent_runner, agent.id),
    )

    async def report() -> None:
        await state.agent_runner.finished(agent.id, turn.finished)

    state.turns.start(turn, after=report)


async def start_notice_turn(state: AppState, session_id: str, text: str) -> bool:
    """Bring an agent's question or report to a session as a turn with source ``agent``.

    False if the session is busy, so the caller can wait and try again. While
    Zero shuts down nothing starts and the notice is dropped.
    """
    if state.stopping:
        return True
    if state.provider is None:
        logger.warning("A notice for session %s was dropped: no usable provider", session_id)
        return True
    if state.turns.busy(session_id):
        return False
    state.turns.start(
        make_turn(state, state.provider, session_id=session_id, text=text, source="agent")
    )
    return True
