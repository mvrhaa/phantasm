"""What the agent runner and the agent tools share; kept apart to avoid an import cycle."""

from dataclasses import dataclass

from zero.store.agents import Agent


class AgentError(Exception):
    """An agent cannot be started or reached; the message says why."""


@dataclass(frozen=True)
class AgentProgress:
    """How far an agent has got: its record, its steps so far and what it last wrote."""

    agent: Agent
    steps: int
    last_output: str | None


@dataclass(frozen=True)
class AgentWait:
    """What waiting for agents found: their messages, or who is still at work."""

    messages: list[str]
    working: list[str]
