"""Starting agents and keeping their status in step with their turns."""

import logging
from collections.abc import Awaitable, Callable

from zero.core.events import EventBus
from zero.protocol.messages import (
    AgentInfo,
    AgentUpdated,
    SessionCreated,
    SessionInfo,
    TurnFinished,
)
from zero.store.agents import Agent, AgentsRepo
from zero.store.sessions import SessionsRepo

NO_REPORT = "(the agent finished without a report)"
RESTARTED = "Zero was restarted before the agent finished."

StartAgentTurn = Callable[[Agent, str], Awaitable[None]]
"""Starts a turn with this text in the agent's session; the runner hears back via ``finished``."""

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """An agent cannot be started or reached; the message says why."""


def agent_info(agent: Agent) -> AgentInfo:
    """What clients are told about an agent."""
    return AgentInfo.model_validate(agent.model_dump())


class AgentRunner:
    """Starts agents in sessions of their own and follows how their turns end.

    ``start_turn`` comes from the server, which knows how to build a turn; it
    must call ``finished`` once the turn is over.
    """

    def __init__(
        self,
        agents: AgentsRepo,
        sessions: SessionsRepo,
        events: EventBus,
        start_turn: StartAgentTurn,
    ) -> None:
        self._agents = agents
        self._sessions = sessions
        self._events = events
        self._start_turn = start_turn

    async def start(self, name: str, task: str, parent_session_id: str) -> Agent:
        """Create the agent's session, store the agent and start its first turn on ``task``.

        Raises ``AgentError`` when an agent tries to start one: agents do not
        start agents.
        """
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

    async def finished(self, agent_id: str, result: TurnFinished | None) -> None:
        """Record how an agent's turn ended: done with its report, failed or stopped.

        Only a working agent changes here; one that was stopped meanwhile stays stopped.
        """
        agent = await self._agents.get(agent_id)
        if agent is None or agent.status != "working":
            return
        if result is None or result.status == "error":
            error = result.error if result is not None else None
            updated = await self._agents.set_status(agent_id, "failed", error or "internal error")
        elif result.status == "cancelled":
            updated = await self._agents.set_status(agent_id, "stopped")
        else:
            updated = await self._agents.set_status(agent_id, "done", result.text or NO_REPORT)
        if updated is not None:
            await self._announce(updated)

    async def recover(self) -> int:
        """Mark agents left at work by an earlier run as failed; returns how many."""
        stale = await self._agents.active()
        for agent in stale:
            await self._agents.set_status(agent.id, "failed", RESTARTED)
        if stale:
            logger.info("%d agents were left unfinished by the last run", len(stale))
        return len(stale)

    async def _announce(self, agent: Agent) -> None:
        await self._events.publish(AgentUpdated(agent=agent_info(agent)))
