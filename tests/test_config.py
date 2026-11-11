"""Tests for zero.config."""

import json
import logging
from pathlib import Path

import pytest

from zero.config import Config, ConfigError, load_config, save_config
from zero.paths import config_path

SPEC_DEFAULTS = {
    "server": {"host": "127.0.0.1", "port": 47400},
    "provider": {"name": "fake", "model": "fake-1", "options": {}},
    "agent": {
        "max_steps": 40,
        "max_steps_computer": 100,
        "turn_timeout_s": 900,
        "history_token_budget": 60000,
    },
    "permissions": {
        "bypass_approvals": False,
        "allowed_roots": ["~"],
        "denied_paths": ["~/.ssh", "~/Library/Keychains"],
        "tool_policies": {
            "read_file": "allow",
            "list_dir": "allow",
            "search_files": "allow",
            "write_file": "ask",
            "edit_file": "ask",
            "run_command": "ask",
            "web_fetch": "allow",
            "memory_save": "allow",
            "memory_search": "allow",
            "memory_delete": "ask",
        },
    },
    "secrets": {},
}


def write_config(data: object) -> Path:
    path = config_path()
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_defaults_match_spec() -> None:
    assert Config().model_dump() == SPEC_DEFAULTS


def test_defaults_are_not_shared_between_instances() -> None:
    first = Config()
    first.permissions.tool_policies["read_file"] = "deny"
    first.permissions.allowed_roots.append("/tmp")

    assert Config().model_dump() == SPEC_DEFAULTS


def test_load_creates_file_with_defaults() -> None:
    path = config_path()
    assert not path.exists()

    config = load_config()

    assert config == Config()
    assert json.loads(path.read_text(encoding="utf-8")) == SPEC_DEFAULTS


def test_load_creates_missing_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.json"

    load_config(path)

    assert path.is_file()


def test_save_then_load_round_trip() -> None:
    config = Config()
    config.server.port = 5000
    config.provider.options = {"temperature": 0.2}
    config.permissions.tool_policies["run_command"] = "deny"
    config.permissions.allowed_roots = ["~/projects"]

    save_config(config)

    assert load_config() == config


def test_save_leaves_no_temporary_file(zero_home: Path) -> None:
    save_config(Config())

    assert [p.name for p in zero_home.iterdir()] == ["config.json"]


def test_partial_file_is_filled_with_defaults() -> None:
    write_config({"server": {"port": 5000}})

    config = load_config()

    assert config.server.port == 5000
    assert config.server.host == "127.0.0.1"
    assert config.agent.max_steps == 40


def test_unknown_keys_warn_but_do_not_fail(caplog: pytest.LogCaptureFixture) -> None:
    write_config({"server": {"port": 5000, "hots": "127.0.0.1"}, "theme": "dark"})

    with caplog.at_level(logging.WARNING, logger="zero.config"):
        config = load_config()

    assert config.server.port == 5000
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any("'theme'" in message for message in warnings)
    assert any("'server.hots'" in message for message in warnings)


def test_unknown_keys_survive_save() -> None:
    write_config({"theme": "dark"})

    save_config(load_config())

    assert json.loads(config_path().read_text(encoding="utf-8"))["theme"] == "dark"


def test_free_form_sections_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    write_config(
        {
            "provider": {"options": {"temperature": 0.2}},
            "permissions": {"tool_policies": {"my_custom_tool": "ask"}},
        }
    )

    with caplog.at_level(logging.WARNING, logger="zero.config"):
        config = load_config()

    assert caplog.records == []
    assert config.provider.options == {"temperature": 0.2}
    assert config.permissions.tool_policies == {"my_custom_tool": "ask"}


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        "[]",
        '{"server": {"port": "not a number"}}',
        '{"agent": "not an object"}',
    ],
)
def test_invalid_file_raises_config_error(text: str) -> None:
    path = config_path()
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="config.json"):
        load_config()
