"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from zero import APP_NAME, __version__
from zero.agents.runner import AgentRunner
from zero.approvals.manager import ApprovalManager
from zero.config import Config
from zero.core.events import EventBus
from zero.core.review import ReviewRunner
from zero.core.rewind import Rewinder
from zero.core.turn_manager import TurnManager
from zero.paths import db_path
from zero.providers.base import LLMProvider
from zero.providers.registry import ProviderNotConfigured, get_provider, small_model
from zero.scheduler.runner import SchedulerRunner
from zero.server.auth import get_or_create_token
from zero.server.connections import ConnectionManager
from zero.server.state import AppState
from zero.server.thinker import make_claude
from zero.server.turns import start_agent_turn, start_notice_turn, start_scheduled_turn
from zero.server.ws import CLOSE_GOING_AWAY
from zero.server.ws import router as ws_router
from zero.store.agents import AgentsRepo
from zero.store.approvals import ApprovalRulesRepo
from zero.store.db import Database
from zero.store.file_changes import FileChangesRepo
from zero.store.journal import JournalRepo
from zero.store.memory import MemoryRepo
from zero.store.messages import MessagesRepo
from zero.store.reviews import ReviewsRepo
from zero.store.schedules import SchedulesRepo
from zero.store.session_memory import SessionMemoryRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.store.todo import TodoRepo
from zero.store.tool_runs import ToolRunsRepo
from zero.store.usage import UsageRepo
from zero.tools.builtin import builtin_registry
from zero.tools.builtin.jobs import JobManager
from zero.tools.builtin.sessions import SessionStores
from zero.tools.client_actions import ActionBroker

APP_ID = "zero"
SHUTDOWN_GRACE_S = 5.0

logger = logging.getLogger(__name__)


def _choose_provider(config: Config) -> LLMProvider | None:
    try:
        return get_provider(config.provider)
    except ProviderNotConfigured as exc:
        logger.error("Provider is not configured: %s", exc)
        return None


async def graceful_stop(state: AppState, grace_s: float = SHUTDOWN_GRACE_S) -> None:
    """Wind the service down so clients see every turn end.

    New messages are refused, the scheduler stops, every turn is cancelled
    and given ``grace_s`` seconds to finish (approvals and client actions it
    waits for end with it), and then every client is closed with 1001.
    Calling it again does nothing.
    """
    if state.stopping:
        return
    state.stopping = True
    logger.info("Shutting down: %d turns running", len(state.turns.active))
    await state.scheduler.stop()
    await state.reviewer.stop()
    await state.agent_runner.close()
    await state.turns.cancel_all(grace_s)
    await state.jobs.stop_all()
    for connection in state.connections.connections:
        await connection.close(CLOSE_GOING_AWAY, f"{APP_NAME} is shutting down")


def create_app(config: Config, provider: LLMProvider | None = None) -> FastAPI:
    """Build the application. Everything it needs is passed in, nothing is global.

    ``provider`` overrides the one named in the config; tests use it to pass
    a scripted provider.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open and migrate the database and start the scheduler; undo it all on shutdown."""
        db = Database(db_path())
        await db.connect()
        try:
            await db.migrate()
            connections = ConnectionManager()
            events = EventBus(connections)
            memory = MemoryRepo(db)
            schedules = SchedulesRepo(db)
            sessions = SessionsRepo(db)
            agents = AgentsRepo(db)
            messages = MessagesRepo(db)
            turns = TurnManager()
            agent_runner = AgentRunner(
                agents=agents,
                sessions=sessions,
                messages=messages,
                events=events,
                turns=turns,
                start_turn=lambda agent, text: start_agent_turn(state, agent, text),
                start_notice=lambda session_id, text: start_notice_turn(state, session_id, text),
                max_agents=config.agent.max_agents,
            )
            await agent_runner.recover()
            jobs = JobManager()
            session_memory = SessionMemoryRepo(db)
            journal = JournalRepo(db)
            summaries = SummariesRepo(db)
            todos = TodoRepo(db)
            file_changes = FileChangesRepo(db)
            await file_changes.prune()
            reviews = ReviewsRepo(db)
            tool_runs = ToolRunsRepo(db)
            usage = UsageRepo(db)
            app.state.zero = state = AppState(
                config=config,
                db=db,
                sessions=sessions,
                messages=messages,
                token=get_or_create_token(),
                connections=connections,
                events=events,
                turns=turns,
                tools=builtin_registry(
                    memory,
                    todos,
                    schedules,
                    agent_runner,
                    jobs,
                    session_memory,
                    journal,
                    SessionStores(sessions, messages, summaries),
                    file_changes,
                ),
                approvals=ApprovalManager(events),
                approval_rules=ApprovalRulesRepo(db),
                tool_runs=tool_runs,
                actions=ActionBroker(connections),
                memory=memory,
                schedules=schedules,
                summaries=summaries,
                usage=usage,
                agents=agents,
                agent_runner=agent_runner,
                scheduler=SchedulerRunner(
                    schedules, lambda schedule: start_scheduled_turn(state, schedule)
                ),
                provider=provider or _choose_provider(config),
                session_memory=session_memory,
                journal=journal,
                reviewer=ReviewRunner(
                    sessions=sessions,
                    messages=messages,
                    memory=memory,
                    journal=journal,
                    reviews=reviews,
                    provider=lambda: state.provider,
                    model=lambda: small_model(config.provider),
                    busy=lambda session_id: state.turns.busy(session_id),
                    enabled=lambda: config.agent.review_conversations,
                    usage=usage,
                ),
                rewinder=Rewinder(
                    sessions=sessions,
                    messages=messages,
                    changes=file_changes,
                    todos=todos,
                    session_memory=session_memory,
                    agents=agents,
                    summaries=summaries,
                    reviews=reviews,
                    tool_runs=tool_runs,
                    busy=lambda session_id: state.turns.busy(session_id),
                    stop_agent=agent_runner.stop,
                ),
                todos=todos,
                jobs=jobs,
                claude=make_claude(config),
            )
            state.scheduler.start()
            state.reviewer.start()
            try:
                yield
            finally:
                await graceful_stop(state)
                await state.turns.shutdown()
                for thinker in (state.provider, state.claude):
                    close = getattr(thinker, "aclose", None)
                    if close is not None:
                        await close()
        finally:
            app.state.zero = None
            await db.close()

    app = FastAPI(
        title=APP_NAME,
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report that this is a running Zero instance. Needs no authorization."""
        return {"app": APP_ID, "version": __version__, "status": "ok"}

    app.include_router(ws_router)
    return app
