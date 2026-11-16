"""Turns and agents a crash cut off go on after the restart."""

import time
from datetime import timedelta

from starlette.testclient import TestClient

from tests.turns import Harness
from zero import clock
from zero.config import Config
from zero.core.turn_manager import TurnManager
from zero.paths import db_path
from zero.providers.base import TextDelta
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.recovery import CONTINUE
from zero.server.state import get_state
from zero.store.agents import AgentsRepo
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.open_turns import OpenTurnsRepo
from zero.store.sessions import SessionsRepo


class Notes:
    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    async def add(self, turn_id: str, session_id: str, source: str, agent_id: str | None) -> None:
        self.seen.append(("add", turn_id, session_id, source))

    async def remove(self, turn_id: str) -> None:
        self.seen.append(("remove", turn_id))


async def test_a_running_turn_is_noted_and_crossed_out(harness: Harness) -> None:
    notes = Notes()
    manager = TurnManager(notes)
    turn = harness.turn(FakeProvider([[TextDelta(text="hi")]]))
    manager.start(turn)
    await manager.wait_idle()
    assert notes.seen == [
        ("add", turn.id, harness.session_id, "desktop"),
        ("remove", turn.id),
    ]


async def test_open_turns_are_taken_once() -> None:
    async with Database(db_path()) as db:
        await db.migrate()
        session = await SessionsRepo(db).create()
        repo = OpenTurnsRepo(db)
        await repo.add("t1", session.id, "desktop", None)
        await repo.add("t2", session.id, "desktop", None)
        await repo.remove("t2")
        taken = await repo.take_all()
        assert [(t.turn_id, t.source) for t in taken] == [("t1", "desktop")]
        assert await repo.take_all() == []


def test_a_restart_takes_up_what_a_crash_cut_off() -> None:
    import asyncio

    async def crashed() -> tuple[str, str, str, str]:
        async with Database(db_path()) as db:
            await db.migrate()
            sessions, agents, repo = SessionsRepo(db), AgentsRepo(db), OpenTurnsRepo(db)
            cut, old, parent, own = [await sessions.create() for _ in range(4)]
            await repo.add("cut-turn", cut.id, "desktop", None)
            await repo.add("old-turn", old.id, "desktop", None)
            await db.conn.execute(
                "UPDATE open_turns SET started_at = ? WHERE turn_id = 'old-turn'",
                (clock.to_iso(clock.now() - timedelta(hours=3)),),
            )
            await db.conn.commit()
            agent = await agents.add("worker", parent.id, own.id, "Sort the files")
            return cut.id, old.id, agent.id, own.id

    cut, old, agent_id, own = asyncio.run(crashed())
    provider = FakeProvider([[TextDelta(text="Going on.")]] * 4)
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (state.turns.active or len(provider.calls) < 2):
            time.sleep(0.05)
        cut_history = client.portal.call(_texts, state.messages, cut)
        old_history = client.portal.call(_texts, state.messages, old)
        agent_history = client.portal.call(_texts, state.messages, own)
        agent = client.portal.call(state.agents.get, agent_id)
    assert cut_history[0] == ("user", CONTINUE, "recovery")
    assert old_history == []
    assert agent_history[0][1] == CONTINUE
    assert agent is not None and agent.status in ("working", "done")


async def _texts(messages: MessagesRepo, session_id: str) -> list[tuple[str, str, str]]:
    stored = await messages.list(session_id, limit=20)
    return [
        (m.role, "".join(getattr(p, "text", "") for p in m.content), m.source or "") for m in stored
    ]
