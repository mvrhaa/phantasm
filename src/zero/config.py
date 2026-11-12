"""Zero configuration: pydantic models, loading and saving ``config.json``."""

import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from zero.paths import config_path

logger = logging.getLogger(__name__)

Policy = Literal["allow", "ask", "deny"]

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

DEFAULT_TOOL_POLICIES: dict[str, Policy] = {
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
}


class ConfigError(Exception):
    """Raised when ``config.json`` cannot be parsed or fails validation."""


class _Section(BaseModel):
    """Base for config models.

    Unknown keys are kept so they can be reported, and assignments are
    validated so a loaded config cannot be changed into an invalid one.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class ServerConfig(_Section):
    """Where the service listens. Only loopback addresses are allowed."""

    host: str = "127.0.0.1"
    port: int = Field(default=47400, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def _host_must_be_loopback(cls, value: str) -> str:
        if value not in LOOPBACK_HOSTS:
            allowed = ", ".join(sorted(LOOPBACK_HOSTS))
            raise ValueError(f"host must be a loopback address ({allowed})")
        return value


class ProviderConfig(_Section):
    """Which LLM provider and model to use.

    ``small_model`` writes session titles and summaries; without it the
    provider's own choice is used, or ``model``. ``history_token_budget`` sets
    how much history the model gets; without it ``agent.history_token_budget``
    applies.
    """

    name: str = "fake"
    model: str = "fake-1"
    small_model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    history_token_budget: int | None = Field(default=None, ge=1000)


class AgentConfig(_Section):
    """Limits of the agent loop."""

    max_steps: int = 40
    max_steps_computer: int = 100
    turn_timeout_s: int = Field(default=900, ge=1)
    history_token_budget: int = 60000
    max_agents: int = Field(default=3, ge=1, le=20)


class PermissionsConfig(_Section):
    """Sandbox roots and per-tool approval policies.

    Paths are stored exactly as written (``~`` included) and are expanded
    by the code that uses them, not when the config is loaded.
    """

    bypass_approvals: bool = False
    allowed_roots: list[str] = Field(default_factory=lambda: ["~"])
    denied_paths: list[str] = Field(default_factory=lambda: ["~/.ssh", "~/Library/Keychains"])
    tool_policies: dict[str, Policy] = Field(default_factory=lambda: dict(DEFAULT_TOOL_POLICIES))

    @field_validator("allowed_roots", "denied_paths")
    @classmethod
    def _paths_must_be_absolute(cls, value: list[str]) -> list[str]:
        if any(not (item.startswith("~") or Path(item).is_absolute()) for item in value):
            raise ValueError("every path must be absolute or start with ~")
        return value


class Config(_Section):
    """Root of ``config.json``."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    secrets: dict[str, Any] = Field(default_factory=dict)

    @field_validator("secrets")
    @classmethod
    def _secrets_must_be_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value:
            raise ValueError("secrets is reserved and must stay empty; API keys belong in .env")
        return value


def _describe_errors(exc: ValidationError) -> str:
    """Summarize validation errors without echoing values from the file."""
    parts = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False):
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts)


def _unknown_keys(model: BaseModel, prefix: str = "") -> list[str]:
    """Return dotted names of keys that no config model declares."""
    keys = [f"{prefix}{name}" for name in (model.model_extra or {})]
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            keys.extend(_unknown_keys(value, f"{prefix}{name}."))
    return keys


def load_config(path: Path | None = None) -> Config:
    """Load the config, creating the file with defaults if it does not exist.

    Unknown keys are logged as warnings and kept. Invalid JSON or values that
    fail validation raise ``ConfigError``; its message names the offending
    keys but never repeats their values.
    """
    path = path or config_path()
    if not path.exists():
        config = Config()
        save_config(config, path)
        return config

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read config file {path}: {type(exc).__name__}") from None
    try:
        config = Config.model_validate_json(text)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file {path}: {_describe_errors(exc)}") from None

    for key in _unknown_keys(config):
        logger.warning("Unknown config key %r in %s", key, path)
    return config


def save_config(config: Config, path: Path | None = None) -> None:
    """Write the config as indented JSON, replacing the file atomically.

    The config is validated again first, so a value changed in place (for
    example inside ``tool_policies``) cannot produce a file that fails to load.
    """
    path = path or config_path()
    try:
        Config.model_validate(config.model_dump())
    except ValidationError as exc:
        raise ConfigError(f"Refusing to save an invalid config: {_describe_errors(exc)}") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
