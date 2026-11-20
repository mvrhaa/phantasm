"""Tests for WebSocket authorization."""

import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from phantasm.config import Config
from phantasm.logging_setup import log_file_path, setup_logging
from phantasm.server.app import create_app
from phantasm.server.state import get_state


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Config())) as test_client:
        yield test_client


@pytest.fixture
def token(client: TestClient) -> str:
    return get_state(client.app).token  # type: ignore[arg-type]


def close_code(client: TestClient, url: str = "/ws", headers: dict[str, str] | None = None) -> int:
    """Connect and return the code the server closes the connection with."""
    with client.websocket_connect(url, headers=headers or {}) as websocket:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            websocket.receive_text()
    return excinfo.value.code


def stays_open(client: TestClient, url: str = "/ws", headers: dict[str, str] | None = None) -> bool:
    """Connect, send something and close from the client side without being kicked."""
    with client.websocket_connect(url, headers=headers or {}) as websocket:
        websocket.send_text("still here")
        websocket.close()
    return True


def test_connection_without_token_is_closed_with_4401(client: TestClient) -> None:
    assert close_code(client) == 4401


@pytest.mark.parametrize(
    "header",
    ["Bearer wrong-token", "Bearer ", "Basic dXNlcjpwYXNz", "wrong-token"],
)
def test_wrong_authorization_header_is_closed_with_4401(header: str, client: TestClient) -> None:
    assert close_code(client, headers={"Authorization": header}) == 4401


def test_wrong_query_token_is_closed_with_4401(client: TestClient) -> None:
    assert close_code(client, "/ws?token=wrong-token") == 4401
    assert close_code(client, "/ws?token=") == 4401


def test_token_in_authorization_header_is_accepted(client: TestClient, token: str) -> None:
    assert stays_open(client, headers={"Authorization": f"Bearer {token}"})


def test_bearer_scheme_is_case_insensitive(client: TestClient, token: str) -> None:
    assert stays_open(client, headers={"Authorization": f"bearer {token}"})


def test_token_in_query_is_accepted(client: TestClient, token: str) -> None:
    assert stays_open(client, f"/ws?token={token}")


def test_header_wins_over_query(client: TestClient, token: str) -> None:
    headers = {"Authorization": "Bearer wrong-token"}

    assert close_code(client, f"/ws?token={token}", headers) == 4401


@pytest.mark.parametrize(
    "origin", ["http://evil.example", "http://localhost:3000", "https://127.0.0.1:47400", ""]
)
def test_browser_origin_is_closed_with_4403(origin: str, client: TestClient, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}", "Origin": origin}

    assert close_code(client, headers=headers) == 4403


def test_origin_is_checked_before_the_token(client: TestClient) -> None:
    assert close_code(client, headers={"Origin": "http://evil.example"}) == 4403


def test_null_origin_is_accepted(client: TestClient, token: str) -> None:
    assert stays_open(client, headers={"Authorization": f"Bearer {token}", "Origin": "null"})


def test_null_origin_still_needs_a_token(client: TestClient) -> None:
    assert close_code(client, headers={"Origin": "null"}) == 4401


def test_health_still_needs_no_token(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


@pytest.mark.usefixtures("clean_root_logger")
def test_token_from_a_logged_url_is_redacted(token: str) -> None:
    path = setup_logging()

    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" [accepted]', "127.0.0.1:5000", f"/ws?token={token}&x=1"
    )
    logging.getLogger("phantasm.test").info("no secrets here")

    text = path.read_text(encoding="utf-8")
    assert token not in text
    assert '"WebSocket /ws?token=***&x=1" [accepted]' in text
    assert "no secrets here" in text


@pytest.mark.usefixtures("clean_root_logger")
def test_rejections_are_logged_without_the_token(client: TestClient) -> None:
    setup_logging()

    close_code(client, "/ws?token=wrong-secret-value")

    text = log_file_path().read_text(encoding="utf-8")
    assert "missing or wrong token" in text
    assert "wrong-secret-value" not in text
