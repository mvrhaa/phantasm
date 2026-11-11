"""Entry point: ``python -m zero`` starts the service."""

import logging
import sys

import uvicorn

from zero import APP_NAME, __version__
from zero.config import ConfigError, load_config
from zero.logging_setup import setup_logging
from zero.paths import ensure_dirs
from zero.server.app import create_app
from zero.server.auth import get_or_create_token

logger = logging.getLogger("zero.main")  # __name__ is "__main__" under ``python -m``


def main() -> int:
    """Prepare directories, logging, config and token, then run the server.

    Returns the process exit code.
    """
    ensure_dirs()
    log_path = setup_logging()

    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    get_or_create_token()  # make sure the token file exists before clients look for it

    host, port = config.server.host, config.server.port
    logger.info(
        "Starting %s %s on %s:%s, logging to %s", APP_NAME, __version__, host, port, log_path
    )
    uvicorn.run(
        create_app(config),
        host=host,
        port=port,
        log_config=None,  # uvicorn logs through our handlers instead of its own
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
