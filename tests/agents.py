"""Helpers for tests with agents: a model that answers by who asks, and a bare runner."""

import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from phantasm.agents.runner import AgentRunner
from phantasm.core.events import EventBus
from phantasm.core.turn_manager import TurnManager
from phantasm.protocol.messages import AgentUpdated, TextPart
from phantasm.providers.base import Message, StreamEvent, ToolCallEvent, ToolSpec
from phantasm.providers.fake import FakeProvider
from phantasm.store.agents import Agent, AgentsRepo
from phantasm.store.db import Database
from phantasm.store.messages import MessagesRepo
from phantasm.store.sessions import SessionsRepo
from tests.turns import Audience, RecordingClient

AGENT_ROLE = "# Your role"


def is_agent(messages: Sequence[Message]) -> bool:
    """True if the request comes from an agent's turn (its system prompt has the role)."""
    system = messages[0] if messages and messages[0].role == "system" else None
    return system is not None and any(
        isinstance(part, TextPart) and AGENT_ROLE in part.text for part in system.content
    )


class RoleProvider(FakeProvider):
    """Plays the ``main`` script for the main conversation and ``agent`` for agents.

    Agents and the main conversation run at the same time, so one shared script
    would be played in an order nobody can predict. Still a FakeProvider, so
    titles and summaries are made without calling it.
    """

    def __init__(self, main: FakeProvider, agent: FakeProvider) -> None:
        super().__init__()
        self.main = main
        self.agent = agent

    async def stream(  # type: ignore[override]
        self, messages: list[Message], tools: list[ToolSpec], model: str, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        target = self.agent if is_agent(messages) else self.main
        async for event in target.stream(messages, tools, model, **kwargs):
            if isinstance(event, ToolCallEvent) and AGENT_ID in event.call.input.values():
                agent_id = last_agent_id(messages)
                filled = {k: agent_id if v == AGENT_ID else v for k, v in event.call.input.items()}
                event = ToolCallEvent(call=event.call.model_copy(update={"input": filled}))
            yield event


class RunnerKit:
    """An agent runner on a real database whose agent turns and notices are only recorded.

    A notice waits like a real one while ``turns`` runs a turn in its session.
    """

    def __init__(self, db: Database, max_agents: int = 3) -> None:
        self.sessions = SessionsRepo(db)
        self.agents = AgentsRepo(db)
        self.messages = MessagesRepo(db)
        self.turns = TurnManager()
        self.client = RecordingClient()
        self.started: list[tuple[str, str]] = []
        self.notices: list[tuple[str, str]] = []

        async def start_turn(agent: Agent, text: str) -> None:
            self.started.append((agent.id, text))

        async def start_notice(session_id: str, text: str) -> bool:
            if self.turns.busy(session_id):
                return False
            self.notices.append((session_id, text))
            return True

        self.runner = AgentRunner(
            agents=self.agents,
            sessions=self.sessions,
            messages=self.messages,
            events=EventBus(Audience([self.client])),
            turns=self.turns,
            start_turn=start_turn,
            start_notice=start_notice,
            max_agents=max_agents,
        )

    @property
    def updates(self) -> list[tuple[str, str | None]]:
        """(status, report) of every agent.updated sent so far."""
        return [
            (e.agent.status, e.agent.report)
            for e in self.client.received
            if isinstance(e, AgentUpdated)
        ]


AGENT_ID = "{agent_id}"
"""Put this in a scripted tool input; RoleProvider swaps in the id of the last agent started."""
_STARTED = re.compile(r"\(id ([0-9a-f-]{36})\)")


def last_agent_id(messages: Sequence[Message]) -> str | None:
    """The id of the most recent agent the conversation heard of."""
    for message in reversed(messages):
        for part in message.content:
            if isinstance(part, TextPart) and (found := _STARTED.findall(part.text)):
                return str(found[-1])
    return None
