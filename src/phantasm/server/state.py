"""Everything a running application depends on, gathered in one object."""

from dataclasses import dataclass, field

from fastapi import FastAPI

from phantasm.agents.runner import AgentRunner
from phantasm.approvals.manager import ApprovalManager
from phantasm.browser import BrowserBridge
from phantasm.config import Config
from phantasm.core.events import EventBus
from phantasm.core.review import ReviewRunner
from phantasm.core.rewind import Rewinder
from phantasm.core.turn_manager import TurnManager
from phantasm.plugins import PluginHost
from phantasm.projects import ProjectBook
from phantasm.providers.base import LLMProvider
from phantasm.scheduler.runner import SchedulerRunner
from phantasm.server.connections import ConnectionManager
from phantasm.skills import SkillBook
from phantasm.store.agents import AgentsRepo
from phantasm.store.approvals import ApprovalRulesRepo
from phantasm.store.db import Database
from phantasm.store.journal import JournalRepo
from phantasm.store.memory import MemoryRepo
from phantasm.store.messages import MessagesRepo
from phantasm.store.schedules import SchedulesRepo
from phantasm.store.session_memory import SessionMemoryRepo
from phantasm.store.sessions import SessionsRepo
from phantasm.store.summaries import SummariesRepo
from phantasm.store.todo import TodoRepo
from phantasm.store.tool_runs import ToolRunsRepo
from phantasm.store.usage import UsageRepo
from phantasm.tools.base import ToolRegistry
from phantasm.tools.builtin.jobs import JobManager
from phantasm.tools.client_actions import ActionBroker
from phantasm.tools.mcp import McpHost


@dataclass
class AppState:
    """Dependencies of one application instance. Built in the lifespan, never global."""

    config: Config
    db: Database
    sessions: SessionsRepo
    messages: MessagesRepo
    token: str
    connections: ConnectionManager
    events: EventBus
    turns: TurnManager
    tools: ToolRegistry
    approvals: ApprovalManager
    approval_rules: ApprovalRulesRepo
    tool_runs: ToolRunsRepo
    actions: ActionBroker
    memory: MemoryRepo
    schedules: SchedulesRepo
    summaries: SummariesRepo
    usage: UsageRepo
    agents: AgentsRepo
    agent_runner: AgentRunner
    scheduler: SchedulerRunner
    provider: LLMProvider | None
    session_memory: SessionMemoryRepo
    journal: JournalRepo
    reviewer: ReviewRunner
    rewinder: Rewinder
    todos: TodoRepo
    jobs: JobManager = field(default_factory=JobManager)
    claude: LLMProvider | None = None
    skills: SkillBook = field(default_factory=SkillBook)
    projects: ProjectBook | None = None
    mcp: McpHost = field(default_factory=McpHost)
    plugins: PluginHost = field(default_factory=PluginHost)
    browser: BrowserBridge = field(default_factory=BrowserBridge)
    # Tools each session loaded on demand (tools_load), kept while the brain runs.
    loaded_tools: dict[str, set[str]] = field(default_factory=dict)
    stopping: bool = False


def get_state(app: FastAPI) -> AppState:
    """Return the state of a started application.

    Raises ``RuntimeError`` before startup and after shutdown.
    """
    state = getattr(app.state, "phantasm", None)
    if not isinstance(state, AppState):
        raise RuntimeError("Application is not started")
    return state
