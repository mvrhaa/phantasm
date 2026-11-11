"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def zero_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ZERO_HOME at a temporary directory so tests never touch real app data."""
    monkeypatch.setenv("ZERO_HOME", str(tmp_path))
    return tmp_path
