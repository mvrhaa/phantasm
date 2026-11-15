"""Everything a running application depends on, gathered in one object."""

from dataclasses import dataclass, field

from fastapi import FastAPI

from zero.agents.runner import AgentRunner
from zero.approvals.manager import ApprovalManager
from zero.config import Config
from zero.core.events import EventBus
from zero.core.turn_manager import TurnManager
from zero.providers.base import LLMProvider
from zero.scheduler.runner import SchedulerRunner
from zero.server.connections import ConnectionManager
from zero.store.agents import AgentsRepo
from zero.store.approvals import ApprovalRulesRepo
from zero.store.db import Database
from zero.store.memory import MemoryRepo
from zero.store.messages import MessagesRepo
from zero.store.schedules import SchedulesRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.store.tool_runs import ToolRunsRepo
from zero.store.usage import UsageRepo
from zero.tools.base import ToolRegistry
from zero.tools.builtin.jobs import JobManager
from zero.tools.client_actions import ActionBroker


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
    jobs: JobManager = field(default_factory=JobManager)
    stopping: bool = False


def get_state(app: FastAPI) -> AppState:
    """Return the state of a started application.

    Raises ``RuntimeError`` before startup and after shutdown.
    """
    state = getattr(app.state, "zero", None)
    if not isinstance(state, AppState):
        raise RuntimeError("Application is not started")
    return state
