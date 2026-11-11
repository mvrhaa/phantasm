"""Agent tools: hand a task to a copy of Zero that works on it in the background."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from zero.agents.runner import AgentError
from zero.store.agents import Agent
from zero.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

AGENT_TOOL_PREFIX = "agent_"  # agents get every tool except these
MAX_NAME_CHARS = 40
MAX_TASK_CHARS = 8000


class AgentControl(Protocol):
    """What the agent tools need from the agent runner."""

    async def start(self, name: str, task: str, parent_session_id: str) -> Agent: ...


class AgentStartInput(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=MAX_NAME_CHARS,
        description="A short name for the agent that says what it does, e.g. 'test runner'",
    )
    task: str = Field(
        min_length=1,
        max_length=MAX_TASK_CHARS,
        description=(
            "Everything the agent needs to know. It sees only this text, not this "
            "conversation, so include paths, goals and what the report should contain."
        ),
    )


class AgentStart(Tool[AgentStartInput]):
    """Start an agent on a task; it reports back when done."""

    name = "agent_start"
    description = (
        "Start an agent: a copy of you with a clean context and the same tools that works "
        "on one task in the background while this conversation goes on. Use it for "
        "self-contained work that takes many steps. Its report comes back here when it is done."
    )
    input_model = AgentStartInput
    default_policy = "allow"

    def __init__(self, control: AgentControl) -> None:
        self._control = control

    def label(self, params: AgentStartInput) -> str:
        return f"agent {params.name}"

    async def run(self, params: AgentStartInput, ctx: ToolContext) -> ToolResult:
        try:
            agent = await self._control.start(params.name, params.task, ctx.session_id)
        except AgentError as exc:
            return error_result(f"The agent could not be started: {exc}.")
        return text_result(
            f"Started agent '{agent.name}' (id {agent.id}). It works in the background; "
            "its report will come to this conversation when it is done."
        )


def agent_tools(control: AgentControl) -> list[Tool[Any]]:
    """Every agent tool, working through ``control``."""
    return [AgentStart(control)]
