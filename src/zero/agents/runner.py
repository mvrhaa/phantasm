"""Running agents: starting them, talking to them, stopping them and hearing back.

An agent works in a session of its own. What it has to tell the session that
started it (a question, its report, its failure) arrives there as a message
with source ``agent`` that starts a turn, as soon as that session is free.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from zero.agents.types import AgentError, AgentProgress
from zero.core.events import EventBus
from zero.core.turn_manager import TurnManager
from zero.protocol.messages import (
    AgentInfo,
    AgentStatus,
    AgentUpdated,
    SessionCreated,
    SessionDeleted,
    SessionInfo,
    TextPart,
    TurnFinished,
)
from zero.store.agents import Agent, AgentsRepo
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo

NO_REPORT = "(the agent finished without a report)"
RESTARTED = "Zero was restarted before the agent finished."
LAST_OUTPUT_CHARS = 1000
STATUS_MESSAGES = 200

StartAgentTurn = Callable[[Agent, str], Awaitable[None]]
"""Starts a turn with this text in the agent's session; it must call ``finished`` after."""
StartNoticeTurn = Callable[[str, str], Awaitable[bool]]
"""Starts a turn in a session with this text from an agent; False if the session is busy."""

logger = logging.getLogger(__name__)


def agent_info(agent: Agent) -> AgentInfo:
    """What clients are told about an agent."""
    return AgentInfo.model_validate(agent.model_dump())


def _who(agent: Agent) -> str:
    return f"Agent '{agent.name}' (id {agent.id})"


class AgentRunner:
    """Starts agents in sessions of their own and carries messages between them and parents.

    ``start_turn`` and ``start_notice`` come from the server, which knows how to
    build turns. ``start_turn`` must call ``finished`` once the agent's turn is over.
    """

    def __init__(
        self,
        *,
        agents: AgentsRepo,
        sessions: SessionsRepo,
        messages: MessagesRepo,
        events: EventBus,
        turns: TurnManager,
        start_turn: StartAgentTurn,
        start_notice: StartNoticeTurn,
    ) -> None:
        self._agents = agents
        self._sessions = sessions
        self._messages = messages
        self._events = events
        self._turns = turns
        self._start_turn = start_turn
        self._start_notice = start_notice
        self._deliveries: set[asyncio.Task[None]] = set()
        self._closed = False

    async def start(self, name: str, task: str, parent_session_id: str) -> Agent:
        """Create the agent's session, store the agent and start its first turn on ``task``."""
        if await self._agents.by_session(parent_session_id) is not None:
            raise AgentError("an agent cannot start agents")
        session = await self._sessions.create(f"Agent: {' '.join(name.split())}")
        try:
            agent = await self._agents.add(name, parent_session_id, session.id, task)
        except ValueError as exc:
            await self._sessions.delete(session.id)
            raise AgentError(str(exc)) from exc
        info = SessionInfo(id=session.id, title=session.title, updated_at=session.updated_at)
        await self._events.publish(SessionCreated(session=info))
        await self._announce(agent)
        logger.info("Agent %s (%s) started by session %s", agent.id, agent.name, parent_session_id)
        await self._start_turn(agent, agent.task)
        return agent

    async def message(self, agent_id: str, text: str, parent_session_id: str) -> Agent:
        """Send the agent a message: an answer, a correction or more work. It gets back to work."""
        agent = await self._own(agent_id, parent_session_id)
        if agent.session_id is None:
            raise AgentError(f"{_who(agent)} was stopped and cannot get messages")
        if self._turns.busy(agent.session_id):
            raise AgentError(f"{_who(agent)} is still working; wait for its report or stop it")
        updated = await self._set(agent, "working")
        await self._start_turn(updated, text)
        return updated

    async def progress(self, agent_id: str, parent_session_id: str) -> AgentProgress:
        """The agent's status, how many steps it took and the last thing it wrote."""
        agent = await self._own(agent_id, parent_session_id)
        if agent.session_id is None:
            return AgentProgress(agent, 0, None)
        stored = await self._messages.list(agent.session_id, limit=STATUS_MESSAGES)
        replies = [m for m in stored if m.role == "assistant"]
        texts = ["".join(p.text for p in m.content if isinstance(p, TextPart)) for m in replies]
        last = next((text.strip() for text in reversed(texts) if text.strip()), None)
        if last is not None and len(last) > LAST_OUTPUT_CHARS:
            last = "…" + last[-(LAST_OUTPUT_CHARS - 1) :]
        return AgentProgress(agent, len(replies), last)

    async def stop(self, agent_id: str, parent_session_id: str) -> Agent:
        """Cancel the agent's work and delete its session; its record stays."""
        agent = await self._own(agent_id, parent_session_id)
        if agent.session_id is None:
            return agent
        if agent.active:
            agent = await self._set(agent, "stopped")  # before cancelling: ``finished`` keeps it
        await self._turns.stop_session(agent.session_id)
        await self._sessions.delete(agent.session_id)
        await self._events.publish(SessionDeleted(session_id=agent.session_id))
        stopped = await self._agents.get(agent.id)
        assert stopped is not None
        await self._announce(stopped)
        logger.info("Agent %s stopped", agent.id)
        return stopped

    async def ask(self, session_id: str, question: str) -> Agent:
        """An agent asks the session that started it; the answer comes as an ``agent_message``."""
        agent = await self._agents.by_session(session_id)
        if agent is None:
            raise AgentError("only agents can ask the conversation that started them")
        asking = await self._set(agent, "asking")
        self._deliver(
            agent.parent_session_id,
            f"{_who(agent)} asks:\n{question.strip()}\n\n"
            "Answer it with agent_message, or ask the user first if only they know.",
        )
        return asking

    async def finished(self, agent_id: str, result: TurnFinished | None) -> None:
        """Record how an agent's turn ended and tell its parent about a report or a failure.

        Only a working agent changes here: one that asked a question keeps
        waiting for the answer, and one that was stopped stays stopped.
        """
        agent = await self._agents.get(agent_id)
        if agent is None or agent.status != "working":
            return
        if result is not None and result.status == "cancelled":
            await self._set(agent, "stopped")
        elif result is None or result.status == "error":
            error = (result.error if result is not None else None) or "internal error"
            failed = await self._set(agent, "failed", error)
            self._deliver(agent.parent_session_id, f"{_who(failed)} failed: {error}")
        else:
            done = await self._set(agent, "done", result.text.strip() or NO_REPORT)
            self._deliver(
                agent.parent_session_id, f"{_who(done)} has finished. Its report:\n{done.report}"
            )

    async def recover(self) -> int:
        """Mark agents left at work by an earlier run as failed; returns how many."""
        stale = await self._agents.active()
        for agent in stale:
            await self._agents.set_status(agent.id, "failed", RESTARTED)
        if stale:
            logger.info("%d agents were left unfinished by the last run", len(stale))
        return len(stale)

    async def close(self) -> None:
        """Drop the messages still waiting for their session; nothing is delivered after this."""
        self._closed = True
        for task in self._deliveries:
            task.cancel()
        await asyncio.gather(*self._deliveries, return_exceptions=True)

    async def _own(self, agent_id: str, parent_session_id: str) -> Agent:
        agent = await self._agents.get(agent_id)
        if agent is None or agent.parent_session_id != parent_session_id:
            raise AgentError(f"this conversation has no agent with id {agent_id}")
        return agent

    async def _set(self, agent: Agent, status: AgentStatus, report: str | None = None) -> Agent:
        updated = await self._agents.set_status(agent.id, status, report)
        assert updated is not None
        await self._announce(updated)
        return updated

    async def _announce(self, agent: Agent) -> None:
        await self._events.publish(AgentUpdated(agent=agent_info(agent)))

    def _deliver(self, session_id: str, text: str) -> None:
        if self._closed:
            return
        task = asyncio.create_task(self._deliver_when_free(session_id, text), name="agent-notice")
        self._deliveries.add(task)
        task.add_done_callback(self._deliveries.discard)

    async def _deliver_when_free(self, session_id: str, text: str) -> None:
        while True:
            await self._turns.wait_free(session_id)
            if await self._sessions.get(session_id) is None:
                return  # the parent session was deleted meanwhile
            if await self._start_notice(session_id, text):
                return
