"""Tests for the ``python -m zero`` entry point. The server itself is never started."""

import json
import stat
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zero import __main__ as entry
from zero.__main__ import PortState, main
from zero.logging_setup import log_file_path
from zero.paths import config_path, token_path

pytestmark = pytest.mark.usefixtures("clean_root_logger")


class FakeUvicorn:
    """Stands in for running the server and records what it would have served, and where."""

    def __init__(self) -> None:
        self.calls: list[tuple[FastAPI, dict[str, Any]]] = []

    def run(self, server: entry.ZeroServer) -> None:
        config = server.config
        settings = {
            "host": config.host,
            "port": config.port,
            "log_config": config.log_config,
            "access_log": config.access_log,
        }
        self.calls.append((config.app, settings))


@pytest.fixture(autouse=True)
def port_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real single-instance probe so these tests never touch the network."""
    monkeypatch.setattr(entry, "probe_port", lambda host, port: PortState.FREE)


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> FakeUvicorn:
    fake = FakeUvicorn()
    monkeypatch.setattr(entry.ZeroServer, "run", lambda server, *args, **kwargs: fake.run(server))
    return fake


def test_main_runs_the_app_on_default_host_and_port(fake_uvicorn: FakeUvicorn) -> None:
    exit_code = main()

    assert exit_code == 0
    assert len(fake_uvicorn.calls) == 1
    app, kwargs = fake_uvicorn.calls[0]
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 47400
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_main_takes_host_and_port_from_config(fake_uvicorn: FakeUvicorn) -> None:
    config_path().write_text(
        json.dumps({"server": {"host": "localhost", "port": 5000}}), encoding="utf-8"
    )

    main()

    _, kwargs = fake_uvicorn.calls[0]
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == 5000


def test_first_run_creates_config_token_and_log(fake_uvicorn: FakeUvicorn) -> None:
    main()

    assert config_path().is_file()
    assert stat.S_IMODE(token_path().stat().st_mode) == 0o600
    assert "Starting Zero" in log_file_path().read_text(encoding="utf-8")


def test_token_is_not_written_to_the_log(fake_uvicorn: FakeUvicorn) -> None:
    main()

    token = token_path().read_text(encoding="utf-8")
    assert token not in log_file_path().read_text(encoding="utf-8")


def test_uvicorn_logs_through_our_handlers_without_access_log(fake_uvicorn: FakeUvicorn) -> None:
    main()

    _, kwargs = fake_uvicorn.calls[0]
    assert kwargs["log_config"] is None
    assert kwargs["access_log"] is False


def test_invalid_config_stops_startup_with_exit_code_1(fake_uvicorn: FakeUvicorn) -> None:
    config_path().write_text(json.dumps({"server": {"host": "0.0.0.0"}}), encoding="utf-8")

    exit_code = main()

    assert exit_code == 1
    assert fake_uvicorn.calls == []
    assert "server.host" in log_file_path().read_text(encoding="utf-8")
