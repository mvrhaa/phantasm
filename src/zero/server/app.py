"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from zero import APP_NAME, __version__
from zero.config import Config
from zero.core.turn import TurnManager
from zero.paths import db_path
from zero.providers.base import LLMProvider
from zero.providers.registry import ProviderNotConfigured, get_provider
from zero.server.auth import get_or_create_token
from zero.server.connections import ConnectionManager
from zero.server.state import AppState
from zero.server.ws import router as ws_router
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo

APP_ID = "zero"

logger = logging.getLogger(__name__)


def _choose_provider(config: Config) -> LLMProvider | None:
    try:
        return get_provider(config.provider)
    except ProviderNotConfigured as exc:
        logger.error("Provider is not configured: %s", exc)
        return None


def create_app(config: Config, provider: LLMProvider | None = None) -> FastAPI:
    """Build the application. Everything it needs is passed in, nothing is global.

    ``provider`` overrides the one named in the config; tests use it to pass
    a scripted provider.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open and migrate the database on startup, close it on shutdown."""
        db = Database(db_path())
        await db.connect()
        try:
            await db.migrate()
            app.state.zero = AppState(
                config=config,
                db=db,
                sessions=SessionsRepo(db),
                messages=MessagesRepo(db),
                token=get_or_create_token(),
                connections=ConnectionManager(),
                turns=TurnManager(),
                provider=provider or _choose_provider(config),
            )
            turns = app.state.zero.turns
            try:
                yield
            finally:
                await turns.shutdown()
        finally:
            app.state.zero = None
            await db.close()

    # The interactive docs and the OpenAPI schema are switched off: the service
    # exposes only /health and the WebSocket.
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
