"""Entry point: ``python -m phantasm`` starts the service."""

import enum
import logging
import socket
import sys

import httpx
import uvicorn

from phantasm import APP_NAME, __version__
from phantasm.config import ConfigError, load_config
from phantasm.logging_setup import setup_logging
from phantasm.paths import config_path, ensure_dirs
from phantasm.server.app import APP_ID, create_app, graceful_stop
from phantasm.server.auth import get_or_create_token
from phantasm.server.state import AppState
from phantasm.tools.base import prune_outputs

PROBE_TIMEOUT_S = 2.0

logger = logging.getLogger("phantasm.main")


class PortState(enum.Enum):
    """What was found on the port the service wants to listen on."""

    FREE = "free"
    PHANTASM_RUNNING = "phantasm_running"
    TAKEN = "taken"


def probe_port(host: str, port: int, *, transport: httpx.BaseTransport | None = None) -> PortState:
    """Ask whatever listens on ``host:port`` whether it is another Phantasm.

    ``transport`` exists so tests can answer the request without a network.
    """
    netloc = f"[{host}]" if ":" in host else host
    url = f"http://{netloc}:{port}/health"
    try:
        with httpx.Client(transport=transport, timeout=PROBE_TIMEOUT_S, trust_env=False) as client:
            response = client.get(url)
    except httpx.ConnectError:
        return PortState.FREE
    except httpx.HTTPError:
        return PortState.TAKEN

    if response.status_code != 200:
        return PortState.TAKEN
    try:
        payload = response.json()
    except ValueError:
        return PortState.TAKEN
    if isinstance(payload, dict) and payload.get("app") == APP_ID:
        return PortState.PHANTASM_RUNNING
    return PortState.TAKEN


class PhantasmServer(uvicorn.Server):
    """uvicorn's server, but Phantasm winds down before uvicorn drops the connections.

    On SIGTERM or SIGINT uvicorn closes every connection first and only then
    runs the app's shutdown, too late to tell clients how their turns ended.
    """

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        state = getattr(self.config.app.state, "phantasm", None)
        if isinstance(state, AppState):
            await graceful_stop(state)
        await super().shutdown(sockets)


def main() -> int:
    """Prepare directories, logging, config and token, then run the server.

    Returns the process exit code: 0 if the service ran or is already running,
    1 if the config is invalid or the port belongs to another program.
    """
    ensure_dirs()
    log_path = setup_logging()
    prune_outputs()

    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    host, port = config.server.host, config.server.port
    state = probe_port(host, port)
    if state is PortState.PHANTASM_RUNNING:
        logger.info("%s is already running on %s:%s, nothing to do", APP_NAME, host, port)
        return 0
    if state is PortState.TAKEN:
        logger.error(
            "Port %s on %s is used by another program. Stop it or change server.port in %s",
            port,
            host,
            config_path(),
        )
        return 1

    get_or_create_token()

    logger.info(
        "Starting %s %s on %s:%s, logging to %s", APP_NAME, __version__, host, port, log_path
    )
    server = PhantasmServer(
        uvicorn.Config(
            create_app(config),
            host=host,
            port=port,
            log_config=None,
            access_log=False,
        )
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
