"""Tests for the application factory and GET /health."""

import pytest
from fastapi.testclient import TestClient

import zero
from zero.config import Config
from zero.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Config()))


def test_health_reports_app_version_and_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"app": "zero", "version": zero.__version__, "status": "ok"}


def test_health_needs_no_authorization(client: TestClient) -> None:
    response = client.get("/health", headers={})

    assert "authorization" not in response.request.headers
    assert response.status_code == 200


def test_health_accepts_only_get(client: TestClient) -> None:
    assert client.post("/health").status_code == 405


@pytest.mark.parametrize("path", ["/", "/docs", "/redoc", "/openapi.json", "/nope"])
def test_nothing_else_is_served(path: str, client: TestClient) -> None:
    assert client.get(path).status_code == 404


def test_each_call_builds_an_independent_app() -> None:
    assert create_app(Config()) is not create_app(Config())
