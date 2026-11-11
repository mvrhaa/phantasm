"""Everything a running application depends on, gathered in one object."""

from dataclasses import dataclass

from fastapi import FastAPI

from zero.approvals.manager import ApprovalManager
from zero.config import Config
from zero.core.events import EventBus
from zero.core.turn_manager import TurnManager
from zero.providers.base import LLMProvider
from zero.scheduler.runner import SchedulerRunner
from zero.server.connections import ConnectionManager
from zero.store.approvals import ApprovalRulesRepo
from zero.store.db import Database
from zero.store.memory import MemoryRepo
from zero.store.messages import MessagesRepo
from zero.store.schedules import SchedulesRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.store.tool_runs import ToolRunsRepo
from zero.tools.base import ToolRegistry
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
    tools: ToolRegistry  # built-in tools, shared by every turn
    approvals: ApprovalManager
    approval_rules: ApprovalRulesRepo
    tool_runs: ToolRunsRepo
    actions: ActionBroker  # client actions waiting for an action.result
    memory: MemoryRepo
    schedules: SchedulesRepo
    summaries: SummariesRepo
    scheduler: SchedulerRunner
    provider: LLMProvider | None  # None when the config names an unusable provider
    stopping: bool = False  # set when the service starts to shut down


def get_state(app: FastAPI) -> AppState:
    """Return the state of a started application.

    Raises ``RuntimeError`` before startup and after shutdown.
    """
    state = getattr(app.state, "zero", None)
    if not isinstance(state, AppState):
        raise RuntimeError("Application is not started")
    return state
