"""FastAPI application factory."""

from fastapi import FastAPI

from zero import APP_NAME, __version__
from zero.config import Config

APP_ID = "zero"


def create_app(config: Config) -> FastAPI:
    """Build the application. Everything it needs is passed in, nothing is global."""
    # The interactive docs and the OpenAPI schema are switched off: the service
    # exposes only /health and the WebSocket.
    app = FastAPI(
        title=APP_NAME,
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = config

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report that this is a running Zero instance. Needs no authorization."""
        return {"app": APP_ID, "version": __version__, "status": "ok"}

    return app
