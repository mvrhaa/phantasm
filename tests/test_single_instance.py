"""Tests for the single-instance check. Requests are answered by a mock transport."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from zero import __main__ as entry
from zero.__main__ import PortState, main, probe_port
from zero.logging_setup import log_file_path
from zero.paths import config_path, token_path

ZERO_HEALTH = {"app": "zero", "version": "0.0.1", "status": "ok"}

Handler = Callable[[httpx.Request], httpx.Response]


def respond(status: int = 200, **kwargs: Any) -> Handler:
    return lambda request: httpx.Response(status, **kwargs)


def fail_with(error: type[httpx.HTTPError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated", request=request)

    return handler


def probe(handler: Handler, host: str = "127.0.0.1", port: int = 47400) -> PortState:
    return probe_port(host, port, transport=httpx.MockTransport(handler))


def test_running_zero_is_recognized() -> None:
    assert probe(respond(json=ZERO_HEALTH)) is PortState.ZERO_RUNNING


def test_refused_connection_means_port_is_free() -> None:
    assert probe(fail_with(httpx.ConnectError)) is PortState.FREE


@pytest.mark.parametrize(
    "handler",
    [
        respond(json={"app": "something-else", "status": "ok"}),
        respond(json={"status": "ok"}),
        respond(json=["zero"]),
        respond(text="<html>hello</html>"),
        respond(404, json=ZERO_HEALTH),
        respond(500),
        respond(302, headers={"location": "http://example.com/"}),
        fail_with(httpx.ReadTimeout),
        fail_with(httpx.RemoteProtocolError),
    ],
    ids=[
        "other-app",
        "no-app-field",
        "json-list",
        "html",
        "not-found",
        "server-error",
        "redirect",
        "timeout",
        "not-http",
    ],
)
def test_anything_else_means_port_is_taken(handler: Handler) -> None:
    assert probe(handler) is PortState.TAKEN


@pytest.mark.parametrize(
    ("host", "expected_url"),
    [
        ("127.0.0.1", "http://127.0.0.1:5000/health"),
        ("localhost", "http://localhost:5000/health"),
        ("::1", "http://[::1]:5000/health"),
    ],
)
def test_probe_requests_health_on_the_given_address(host: str, expected_url: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ZERO_HEALTH)

    probe(handler, host=host, port=5000)

    assert [request.method for request in seen] == ["GET"]
    assert str(seen[0].url) == expected_url


def test_probe_ignores_proxy_settings_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:9")

    assert probe(respond(json=ZERO_HEALTH)) is PortState.ZERO_RUNNING


class FakeUvicorn:
    """Stands in for running the server; records only that it was started."""

    def __init__(self) -> None:
        self.started = False

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.started = True


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> FakeUvicorn:
    fake = FakeUvicorn()
    monkeypatch.setattr(entry.ZeroServer, "run", fake)
    return fake


def set_port_state(monkeypatch: pytest.MonkeyPatch, state: PortState) -> list[tuple[str, int]]:
    probed: list[tuple[str, int]] = []

    def fake_probe(host: str, port: int) -> PortState:
        probed.append((host, port))
        return state

    monkeypatch.setattr(entry, "probe_port", fake_probe)
    return probed


@pytest.mark.usefixtures("clean_root_logger")
def test_second_instance_exits_with_code_0(
    monkeypatch: pytest.MonkeyPatch, fake_uvicorn: FakeUvicorn
) -> None:
    set_port_state(monkeypatch, PortState.ZERO_RUNNING)

    exit_code = main()

    assert exit_code == 0
    assert fake_uvicorn.started is False
    assert "already running on 127.0.0.1:47400" in log_file_path().read_text(encoding="utf-8")
    assert not token_path().exists()


@pytest.mark.usefixtures("clean_root_logger")
def test_port_taken_by_another_program_exits_with_code_1(
    monkeypatch: pytest.MonkeyPatch, fake_uvicorn: FakeUvicorn
) -> None:
    set_port_state(monkeypatch, PortState.TAKEN)

    exit_code = main()

    log = log_file_path().read_text(encoding="utf-8")
    assert exit_code == 1
    assert fake_uvicorn.started is False
    assert "Port 47400 on 127.0.0.1 is used by another program" in log
    assert str(config_path()) in log


@pytest.mark.usefixtures("clean_root_logger")
def test_free_port_starts_the_server(
    monkeypatch: pytest.MonkeyPatch, fake_uvicorn: FakeUvicorn
) -> None:
    set_port_state(monkeypatch, PortState.FREE)

    exit_code = main()

    assert exit_code == 0
    assert fake_uvicorn.started is True


@pytest.mark.usefixtures("clean_root_logger")
def test_main_probes_host_and_port_from_config(
    monkeypatch: pytest.MonkeyPatch, fake_uvicorn: FakeUvicorn
) -> None:
    config_path().write_text(
        json.dumps({"server": {"host": "localhost", "port": 5000}}), encoding="utf-8"
    )
    probed = set_port_state(monkeypatch, PortState.FREE)

    main()

    assert probed == [("localhost", 5000)]
