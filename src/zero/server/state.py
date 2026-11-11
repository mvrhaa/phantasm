"""Everything a running application depends on, gathered in one object."""

from dataclasses import dataclass

from fastapi import FastAPI

from zero.config import Config
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo


@dataclass
class AppState:
    """Dependencies of one application instance. Built in the lifespan, never global."""

    config: Config
    db: Database
    sessions: SessionsRepo
    messages: MessagesRepo
    token: str


def get_state(app: FastAPI) -> AppState:
    """Return the state of a started application.

    Raises ``RuntimeError`` before startup and after shutdown.
    """
    state = getattr(app.state, "zero", None)
    if not isinstance(state, AppState):
        raise RuntimeError("Application is not started")
    return state
