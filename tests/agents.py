"""Helpers for tests with agents: a model whose answers depend on who is asking."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from zero.protocol.messages import TextPart
from zero.providers.base import Message, StreamEvent, ToolSpec
from zero.providers.fake import FakeProvider

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
            yield event
