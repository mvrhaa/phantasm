"""Agent tools: hand a task to a copy of Phantasm, talk to it, check on it, stop it.

The ``agent_*`` tools belong to the main conversation; agents get every tool
except those, plus ``ask_parent`` to put a question to the conversation that
started them.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from phantasm.agents.types import AgentError, AgentProgress, AgentWait
from phantasm.core.cancellation import CancelToken
from phantasm.store.agents import Agent
from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

AGENT_TOOL_PREFIX = "agent_"
MAX_NAME_CHARS = 40
MAX_TASK_CHARS = 8000
MAX_MESSAGE_CHARS = 8000


class AgentControl(Protocol):
    """What the agent tools need from the agent runner."""

    async def start(
        self,
        name: str,
        task: str,
        parent_session_id: str,
        verify: bool = False,
        claude: bool = False,
    ) -> Agent: ...

    async def message(self, agent_id: str, text: str, parent_session_id: str) -> Agent: ...

    async def progress(self, agent_id: str, parent_session_id: str) -> AgentProgress: ...

    async def stop(self, agent_id: str, parent_session_id: str) -> Agent: ...

    async def ask(self, session_id: str, question: str) -> Agent: ...

    async def note_progress(self, session_id: str, note: str) -> Agent: ...

    async def wait(
        self,
        parent_session_id: str,
        agent_ids: Sequence[str],
        timeout_s: float,
        cancel: CancelToken,
    ) -> AgentWait: ...


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
    verify: bool = Field(
        default=False,
        description=(
            "A skeptical reviewer checks the work before the report comes back and sends the "
            "agent back to fix what fails. For work with a checkable outcome: code, files."
        ),
    )
    claude: bool = Field(
        default=False,
        description=(
            "The agent thinks through the user's own Claude Code (Claude, on their Claude "
            "plan) with your tools: for code and hard thinking, or when the user asks for Claude."
        ),
    )


class AgentWaitInput(BaseModel):
    ids: list[str] = Field(
        default_factory=list, description="Agents to wait for (default: all of this conversation)"
    )
    timeout_s: int = Field(default=300, ge=1, le=1800, description="Most seconds to wait")


class AgentProgressInput(BaseModel):
    note: str = Field(
        min_length=1, max_length=2000, description="What you did and what comes next, briefly"
    )


class AgentIdInput(BaseModel):
    id: str = Field(min_length=1, description="Id of the agent, as agent_start returned it")


class AgentMessageInput(AgentIdInput):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, description="The message")


class AskParentInput(BaseModel):
    question: str = Field(
        min_length=1, max_length=MAX_MESSAGE_CHARS, description="The question, self-contained"
    )


class _AgentTool[InputT: BaseModel](Tool[InputT]):
    default_policy = "allow"

    def __init__(self, control: AgentControl) -> None:
        self._control = control


class AgentStart(_AgentTool[AgentStartInput]):
    """Start an agent on a task; it reports back when done."""

    name = "agent_start"

    core = False
    description = (
        "Start an agent: a copy of you with a clean context and the same tools that works "
        "on one task in the background while this conversation goes on. Use it for "
        "self-contained work that takes many steps. Its questions and its report come back "
        "here as messages."
    )
    input_model = AgentStartInput

    def label(self, params: AgentStartInput) -> str:
        return f"agent {params.name}"

    async def run(self, params: AgentStartInput, ctx: ToolContext) -> ToolResult:
        try:
            agent = await self._control.start(
                params.name, params.task, ctx.session_id, params.verify, params.claude
            )
        except AgentError as exc:
            return error_result(f"The agent could not be started: {exc}.")
        return text_result(
            f"Started agent '{agent.name}' (id {agent.id}). It works in the background; "
            "its report will come to this conversation when it is done."
        )


class AgentMessage(_AgentTool[AgentMessageInput]):
    """Send an agent an answer, a correction or more work."""

    name = "agent_message"

    core = False
    description = (
        "Send a message to an agent of this conversation: the answer to its question, a "
        "correction or more work. It gets back to work and reports again when done."
    )
    input_model = AgentMessageInput

    def label(self, params: AgentMessageInput) -> str:
        return f"message to agent {params.id[:8]}"

    async def run(self, params: AgentMessageInput, ctx: ToolContext) -> ToolResult:
        try:
            agent = await self._control.message(params.id, params.text, ctx.session_id)
        except AgentError as exc:
            return error_result(f"The message was not sent: {exc}.")
        return text_result(f"Sent to agent '{agent.name}'. It is working on it.")


class AgentStatusTool(_AgentTool[AgentIdInput]):
    """How an agent is doing."""

    name = "agent_status"

    core = False
    description = (
        "Check on an agent of this conversation: its status, how many steps it took, the "
        "last thing it wrote and its report, if any."
    )
    input_model = AgentIdInput

    def label(self, params: AgentIdInput) -> str:
        return f"status of agent {params.id[:8]}"

    async def run(self, params: AgentIdInput, ctx: ToolContext) -> ToolResult:
        try:
            progress = await self._control.progress(params.id, ctx.session_id)
        except AgentError as exc:
            return error_result(f"No status: {exc}.")
        agent = progress.agent
        lines = [
            f"Agent '{agent.name}' (id {agent.id}): {agent.status}",
            f"Task: {agent.task}",
            f"Steps so far: {progress.steps}",
        ]
        if agent.progress:
            lines.append(f"Its latest progress note: {agent.progress}")
        if progress.last_output:
            lines.append(f"Last output:\n{progress.last_output}")
        if agent.report:
            lines.append(f"Report:\n{agent.report}")
        return text_result("\n".join(lines), summary=agent.status)


class AgentStop(_AgentTool[AgentIdInput]):
    """Stop an agent and delete its session."""

    name = "agent_stop"

    core = False
    description = (
        "Stop an agent of this conversation: its work is cancelled and its session deleted. "
        "Its status and report can still be checked."
    )
    input_model = AgentIdInput

    def label(self, params: AgentIdInput) -> str:
        return f"stop agent {params.id[:8]}"

    async def run(self, params: AgentIdInput, ctx: ToolContext) -> ToolResult:
        try:
            agent = await self._control.stop(params.id, ctx.session_id)
        except AgentError as exc:
            return error_result(f"Nothing was stopped: {exc}.")
        return text_result(f"Agent '{agent.name}' is stopped ({agent.status}).")


class AskParent(_AgentTool[AskParentInput]):
    """An agent's question to the conversation that started it."""

    name = "ask_parent"
    description = (
        "Ask the conversation that started you something only it or the user knows. "
        "Then end your turn: the answer comes as your next message."
    )
    input_model = AskParentInput

    def label(self, params: AskParentInput) -> str:
        return "question to the main conversation"

    async def run(self, params: AskParentInput, ctx: ToolContext) -> ToolResult:
        try:
            await self._control.ask(ctx.session_id, params.question)
        except AgentError as exc:
            return error_result(f"The question was not sent: {exc}.")
        return text_result(
            "Your question was sent. End your turn now; the answer will come as your next message."
        )


class AgentWaitTool(_AgentTool[AgentWaitInput]):
    """Wait for the agents' reports inside this turn."""

    name = "agent_wait"

    core = False
    description = (
        "Wait for the reports or questions of this conversation's agents and get them here, "
        "instead of ending your turn. What this returns does not come again as a message."
    )
    input_model = AgentWaitInput

    def label(self, params: AgentWaitInput) -> str:
        return "waiting for agents"

    async def run(self, params: AgentWaitInput, ctx: ToolContext) -> ToolResult:
        found = await self._control.wait(ctx.session_id, params.ids, params.timeout_s, ctx.cancel)
        if found.messages:
            return text_result("\n\n".join(found.messages))
        if found.working:
            return text_result(
                f"No reports yet after {params.timeout_s} s. Still working: "
                + ", ".join(found.working)
                + ". Their reports come here as messages when they are done."
            )
        return text_result("None of these agents is working, and no report is waiting.")


class AgentProgressTool(_AgentTool[AgentProgressInput]):
    """An agent's note of how far it got."""

    name = "report_progress"
    description = (
        "Note how far you got: what you did and what comes next. The conversation that "
        "started you sees your latest note. Use it after each meaningful part of long work."
    )
    input_model = AgentProgressInput

    def label(self, params: AgentProgressInput) -> str:
        return "progress note"

    async def run(self, params: AgentProgressInput, ctx: ToolContext) -> ToolResult:
        try:
            await self._control.note_progress(ctx.session_id, params.note)
        except AgentError as exc:
            return error_result(f"Not recorded: {exc}.")
        return text_result("Progress recorded.")


def agent_tools(control: AgentControl) -> list[Tool[Any]]:
    """The tools the main conversation uses to run agents, working through ``control``."""
    return [
        AgentStart(control),
        AgentMessage(control),
        AgentStatusTool(control),
        AgentWaitTool(control),
        AgentStop(control),
    ]


def agent_side_tools(control: AgentControl) -> list[Tool[Any]]:
    """The tools only agents get."""
    return [AskParent(control), AgentProgressTool(control)]
