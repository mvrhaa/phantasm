"""Tests for zero.config."""

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from zero.config import Config, ConfigError, ServerConfig, load_config, save_config
from zero.paths import config_path

SPEC_DEFAULTS = {
    "server": {"host": "127.0.0.1", "port": 47400},
    "provider": {
        "name": "fake",
        "model": "fake-1",
        "small_model": None,
        "options": {},
        "history_token_budget": None,
    },
    "claude_code": {"active": False, "model": "sonnet", "command": "claude"},
    "agent": {
        "max_steps": 40,
        "max_steps_computer": 100,
        "turn_timeout_s": 900,
        "history_token_budget": 60000,
        "max_agents": 3,
        "defer_tools": True,
        "resume_after_restart": True,
        "review_conversations": True,
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


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert ServerConfig(host=host).host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com", ""])
def test_non_loopback_host_is_rejected(host: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        ServerConfig(host=host)


def test_non_loopback_host_in_file_raises_config_error() -> None:
    write_config({"server": {"host": "0.0.0.0"}})

    with pytest.raises(ConfigError, match="server.host"):
        load_config()


def test_host_cannot_be_changed_to_non_loopback_after_load() -> None:
    config = load_config()

    with pytest.raises(ValidationError, match="loopback"):
        config.server.host = "0.0.0.0"

    assert config.server.host == "127.0.0.1"


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_port_out_of_range_is_rejected(port: int) -> None:
    with pytest.raises(ValidationError):
        ServerConfig(port=port)


@pytest.mark.parametrize("policy", ["allow", "ask", "deny"])
def test_known_policies_are_accepted(policy: str) -> None:
    write_config({"permissions": {"tool_policies": {"run_command": policy}}})

    assert load_config().permissions.tool_policies == {"run_command": policy}


@pytest.mark.parametrize("policy", ["maybe", "ALLOW", "", None, True])
def test_unknown_policy_raises_config_error(policy: object) -> None:
    write_config({"permissions": {"tool_policies": {"run_command": policy}}})

    with pytest.raises(ConfigError, match="permissions.tool_policies.run_command"):
        load_config()


@pytest.mark.parametrize("key", ["allowed_roots", "denied_paths"])
def test_relative_sandbox_path_raises_config_error(key: str) -> None:
    write_config({"permissions": {key: ["~/ok", "/also/ok", "projects/secret"]}})

    with pytest.raises(ConfigError, match=f"permissions.{key}: .*absolute or start with ~") as err:
        load_config()

    assert "projects/secret" not in str(err.value)


def test_paths_keep_tilde_after_load_and_save() -> None:
    write_config({"permissions": {"allowed_roots": ["~/projects"], "denied_paths": ["~/.aws"]}})

    config = load_config()
    save_config(config)

    assert config.permissions.allowed_roots == ["~/projects"]
    assert config.permissions.denied_paths == ["~/.aws"]
    saved = json.loads(config_path().read_text(encoding="utf-8"))
    assert saved["permissions"]["allowed_roots"] == ["~/projects"]
    assert saved["permissions"]["denied_paths"] == ["~/.aws"]


def test_non_empty_secrets_are_rejected_without_leaking_values() -> None:
    write_config({"secrets": {"some_provider": "sk-very-secret"}})

    with pytest.raises(ConfigError, match="secrets") as excinfo:
        load_config()

    assert "sk-very-secret" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_config_error_does_not_echo_invalid_values() -> None:
    write_config({"server": {"port": "hunter2"}})

    with pytest.raises(ConfigError, match="server.port") as excinfo:
        load_config()

    assert "hunter2" not in str(excinfo.value)


def test_save_refuses_a_config_made_invalid_in_place() -> None:
    config = load_config()
    before = config_path().read_text(encoding="utf-8")
    config.permissions.tool_policies["run_command"] = "sometimes"  # type: ignore[assignment]

    with pytest.raises(ConfigError, match="permissions.tool_policies.run_command"):
        save_config(config)

    assert config_path().read_text(encoding="utf-8") == before
    assert load_config().permissions.tool_policies["run_command"] == "ask"
