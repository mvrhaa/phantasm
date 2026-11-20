"""Choosing who thinks in the user's turns (``provider.set``)."""

import logging

from phantasm.config import ConfigError, save_config
from phantasm.core.events import ReplyTo
from phantasm.protocol.errors import ErrorCode
from phantasm.protocol.messages import Error, ProviderChanged, ProviderSet
from phantasm.server.connections import Connection
from phantasm.server.state import AppState
from phantasm.server.thinker import make_claude, thinker_info

logger = logging.getLogger(__name__)


async def handle_provider_set(
    connection: Connection, message: ProviderSet, state: AppState
) -> None:
    """Turn the user's Claude Code on or off for the turns that start from now on.

    The choice is saved in ``config.json``; every client hears ``provider.changed``.
    """
    if message.claude_code and state.claude is None:
        state.claude = make_claude(state.config)
        if state.claude is None:
            await connection.send(
                Error(
                    code=ErrorCode.PROVIDER_NOT_CONFIGURED,
                    message="Claude Code is not installed: the claude command was not found",
                    reply_to=message.id,
                )
            )
            return
    state.config.claude_code.active = message.claude_code
    try:
        save_config(state.config)
    except (ConfigError, OSError) as exc:
        logger.warning("The thinker choice was not saved: %s", type(exc).__name__)
    await state.events.publish(
        ProviderChanged(thinker=thinker_info(state.config)),
        reply=ReplyTo(connection.id, message.id),
    )
