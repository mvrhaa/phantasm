"""Tests of API keys: kept only in .env, never shown, never reachable by the model."""

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from tests.turns import FuncTool, registry, text_of
from zero import cli, secrets
from zero.config import Config, ConfigError, load_config
from zero.core.cancellation import CancelToken
from zero.logging_setup import log_file_path, setup_logging
from zero.paths import config_path
from zero.providers.base import ToolCall
from zero.secrets import api_key, env_file, mask_secrets, read_env_file
from zero.tools.base import ToolContext
from zero.tools.permissions import PathDenied, PathGuard

KEY = "mk-test-0123456789abcdef"


def write_env(text: str, mode: int = 0o600) -> Path:
    path = env_file()
    path.write_text(text)
    path.chmod(mode)
    os.utime(path)
    return path


def test_tests_never_see_the_real_env_file(zero_home: Path) -> None:
    assert env_file().parent == zero_home.parent
    assert env_file() != Path(secrets.__file__).resolve().parents[2] / ".env"


def test_without_the_override_the_file_is_at_the_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(secrets.ENV_FILE_VAR)

    root = Path(secrets.__file__).resolve().parents[2]
    assert env_file() == root / ".env"
    assert (root / "pyproject.toml").exists()


def test_the_file_format(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# keys\n"
        "\n"
        "MISTRAL_API_KEY = abc123\n"
        'export QUOTED="with spaces"\n'
        "SINGLE='x'\n"
        "EMPTY=\n"
        "not a pair\n"
        "URL=https://a.b/?q=1\n"
    )

    assert read_env_file(path) == {
        "MISTRAL_API_KEY": "abc123",
        "QUOTED": "with spaces",
        "SINGLE": "x",
        "EMPTY": "",
        "URL": "https://a.b/?q=1",
    }
    assert read_env_file(tmp_path / "missing.env") == {}


def test_the_key_comes_from_the_file_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "from-the-shell-environment")

    assert api_key("mistral") is None
    write_env("MISTRAL_API_KEY=\n")
    assert api_key("mistral") is None
    write_env(f"MISTRAL_API_KEY={KEY}\n")
    assert api_key("mistral") == KEY
    assert api_key("fake") is None


def test_values_from_the_file_are_masked() -> None:
    write_env(f"MISTRAL_API_KEY={KEY}\nSHORT=abc\n")

    assert mask_secrets(f"Bearer {KEY} and abc") == "Bearer *** and abc"
    write_env("MISTRAL_API_KEY=another-key-value\n")
    assert mask_secrets(f"{KEY} another-key-value") == f"{KEY} ***"


@pytest.mark.usefixtures("clean_root_logger")
def test_the_key_never_reaches_the_log() -> None:
    write_env(f"MISTRAL_API_KEY={KEY}\n")
    setup_logging()

    logging.getLogger("zero.test").warning("request with header Bearer %s failed", KEY)

    text = log_file_path().read_text(encoding="utf-8")
    assert KEY not in text
    assert "request with header Bearer *** failed" in text


def context() -> ToolContext:
    return ToolContext(
        session_id="s",
        turn_id="t",
        client_id=None,
        cancel=CancelToken(),
        config=Config(),
        paths=PathGuard(["/"], []),
    )


async def test_tool_output_and_errors_carry_no_key() -> None:
    write_env(f"MISTRAL_API_KEY={KEY}\n")

    async def leaky(params: Any, ctx: ToolContext) -> str:
        return f"MISTRAL_API_KEY={KEY}"

    async def broken(params: Any, ctx: ToolContext) -> str:
        raise RuntimeError(f"bad key {KEY}")

    tools = registry(FuncTool("leaky", leaky), FuncTool("broken", broken))
    output = await tools.execute(ToolCall(id="1", name="leaky"), context())
    error = await tools.execute(ToolCall(id="2", name="broken"), context())

    assert text_of(output) == "MISTRAL_API_KEY=***"
    assert text_of(error) == "Tool 'broken' failed: bad key ***"


def test_the_env_file_is_off_limits_to_tools(zero_home: Path) -> None:
    path = write_env(f"MISTRAL_API_KEY={KEY}\n")
    config = Config()
    config.permissions.allowed_roots = [str(zero_home.parent)]
    config.permissions.denied_paths = []
    guard = PathGuard.from_config(config.permissions)

    with pytest.raises(PathDenied, match="protected location"):
        guard.check(path)
    with pytest.raises(PathDenied):
        guard.check(str(path).upper())
    assert guard.check(zero_home / "notes.txt") == (zero_home / "notes.txt").resolve()


def test_the_config_has_no_place_for_keys() -> None:
    config_path().write_text('{"secrets": {"mistral": "x"}}')

    with pytest.raises(ConfigError, match="API keys belong in .env"):
        load_config()


def doctor_line(capsys: pytest.CaptureFixture[str], provider: str) -> str:
    config_path().write_text(f'{{"provider": {{"name": "{provider}"}}}}')
    cli.main(["doctor"], probe=lambda host, port: cli.PortState.FREE)
    out = capsys.readouterr().out
    assert KEY not in out
    line = next(line for line in out.splitlines() if line.split()[1] == "key")
    return " ".join(line.split())


def test_doctor_on_the_fake_provider_needs_no_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert doctor_line(capsys, "fake") == "✓ key none needed for the fake provider"


def test_doctor_reports_a_missing_or_present_key(capsys: pytest.CaptureFixture[str]) -> None:
    missing = doctor_line(capsys, "mistral")
    path = write_env(f"MISTRAL_API_KEY={KEY}\n")
    found = doctor_line(capsys, "mistral")

    assert missing == f"✗ key MISTRAL_API_KEY is missing in {env_file()}"
    assert found == f"✓ key MISTRAL_API_KEY found in {path}"


def test_doctor_names_the_key_of_the_chosen_preset(capsys: pytest.CaptureFixture[str]) -> None:
    write_env("GEMINI_API_KEY=AIza-test-0123456789\n")

    assert doctor_line(capsys, "google") == f"✓ key GEMINI_API_KEY found in {env_file()}"
    assert doctor_line(capsys, "ollama") == "✓ key none needed for the ollama provider"


def test_doctor_wants_the_file_private(capsys: pytest.CaptureFixture[str]) -> None:
    path = write_env(f"MISTRAL_API_KEY={KEY}\n", mode=0o644)

    assert doctor_line(capsys, "mistral") == (
        f"✗ key {path} can be read by others; run chmod 600 on it"
    )
