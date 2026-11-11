"""Smoke test: the package imports and the test environment is isolated."""

import os
from importlib.metadata import version
from pathlib import Path

import zero


def test_smoke(zero_home: Path) -> None:
    assert zero.APP_NAME == "Zero"
    assert zero.__version__ == version("zero")
    assert os.environ["ZERO_HOME"] == str(zero_home)
    assert zero_home.is_dir()
