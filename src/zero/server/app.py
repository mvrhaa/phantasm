"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from zero import APP_NAME, __version__
from zero.config import Config
from zero.paths import db_path
from zero.server.auth import get_or_create_token
from zero.server.state import AppState
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo

APP_ID = "zero"


def create_app(config: Config) -> FastAPI:
    """Build the application. Everything it needs is passed in, nothing is global."""

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
            )
            yield
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

    return app
