"""Smoke test: the package imports and the test environment is isolated."""

import os
from importlib.metadata import version
from pathlib import Path

import phantasm


def test_smoke(phantasm_home: Path) -> None:
    assert phantasm.APP_NAME == "Phantasm"
    assert phantasm.__version__ == version("phantasm")
    assert os.environ["PHANTASM_HOME"] == str(phantasm_home)
    assert phantasm_home.is_dir()
