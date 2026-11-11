"""Tests for the application lifespan and AppState."""

import stat

import pytest
from fastapi.testclient import TestClient

from zero.config import Config
from zero.paths import db_path, token_path
from zero.protocol.messages import TextPart
from zero.server.app import create_app
from zero.server.state import AppState, get_state
from zero.store.db import MIGRATIONS_DIR, MigrationError


def test_startup_opens_a_migrated_database_in_zero_home() -> None:
    app = create_app(Config())
    assert not db_path().exists()

    with TestClient(app) as client:
        state = get_state(app)

        assert db_path().is_file()
        assert state.db.path == db_path()
        latest = max(int(path.name.split("_")[0]) for path in MIGRATIONS_DIR.glob("*.sql"))
        assert client.portal.call(state.db.version) == latest
        assert client.get("/health").status_code == 200


def test_state_holds_the_config_and_the_token() -> None:
    config = Config()
    app = create_app(config)

    with TestClient(app):
        state = get_state(app)

        assert isinstance(state, AppState)
        assert state.config is config
        assert state.token == token_path().read_text(encoding="utf-8")
        assert stat.S_IMODE(token_path().stat().st_mode) == 0o600


def test_repositories_share_the_database() -> None:
    app = create_app(Config())

    with TestClient(app) as client:
        state = get_state(app)
        session = client.portal.call(state.sessions.create, "Plans")
        client.portal.call(state.messages.append, session.id, "user", [TextPart(text="hi")])
        messages = client.portal.call(state.messages.list, session.id)

        assert client.portal.call(state.sessions.get, session.id) == session
        assert [m.content for m in messages] == [[TextPart(text="hi")]]


def test_state_is_unavailable_before_startup_and_after_shutdown() -> None:
    app = create_app(Config())
    with pytest.raises(RuntimeError, match="not started"):
        get_state(app)

    with TestClient(app):
        db = get_state(app).db

    with pytest.raises(RuntimeError, match="not started"):
        get_state(app)
    with pytest.raises(RuntimeError, match="not connected"):
        _ = db.conn


def test_data_survives_a_restart() -> None:
    first = create_app(Config())
    with TestClient(first) as client:
        session = client.portal.call(get_state(first).sessions.create, "Plans")
        token = get_state(first).token

    second = create_app(Config())
    with TestClient(second) as client:
        state = get_state(second)
        assert client.portal.call(state.sessions.get, session.id) == session
        assert state.token == token


def test_two_apps_have_separate_states() -> None:
    first, second = create_app(Config()), create_app(Config())

    with TestClient(first), TestClient(second):
        assert get_state(first) is not get_state(second)
        assert get_state(first).db is not get_state(second).db


def test_health_works_without_startup() -> None:
    client = TestClient(create_app(Config()))

    assert client.get("/health").status_code == 200
    assert not db_path().exists()


def test_failed_migration_stops_startup_and_closes_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_migrate(self: object) -> list[int]:
        raise MigrationError("boom")

    monkeypatch.setattr("zero.store.db.Database.migrate", broken_migrate)
    app = create_app(Config())

    with pytest.raises(MigrationError, match="boom"):
        with TestClient(app):
            pass

    with pytest.raises(RuntimeError, match="not started"):
        get_state(app)
    assert db_path().is_file()
