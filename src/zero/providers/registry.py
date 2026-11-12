"""Choosing the provider named in the config."""

from typing import Any

from zero.config import ProviderConfig
from zero.protocol.errors import ErrorCode
from zero.providers.base import LLMProvider
from zero.providers.chat_completions import ChatCompletionsProvider
from zero.providers.fake import FakeProvider
from zero.providers.presets import PRESETS, Preset
from zero.secrets import api_key, env_file

PRESET_OPTIONS = {"images", "min_interval_s"}


class ProviderNotConfigured(Exception):
    """The config names a provider that does not exist, gives it bad options,
    or its API key is missing."""

    code = ErrorCode.PROVIDER_NOT_CONFIGURED


def get_provider(config: ProviderConfig) -> LLMProvider:
    """Build the provider for ``config.name``, passing ``config.options`` to it."""
    if config.name == "fake":
        try:
            return FakeProvider(**config.options)
        except TypeError as exc:
            raise ProviderNotConfigured(f"bad options for the fake provider: {exc}") from exc
    preset = PRESETS.get(config.name)
    if preset is None:
        raise ProviderNotConfigured(f"unknown provider {config.name!r}")
    return _preset_provider(preset, config.options)


def _preset_provider(preset: Preset, options: dict[str, Any]) -> ChatCompletionsProvider:
    unknown = sorted(set(options) - PRESET_OPTIONS)
    if unknown:
        raise ProviderNotConfigured(
            f"unknown options for the {preset.name} provider: {', '.join(unknown)}"
        )
    images = options.get("images")
    if images is not None and not isinstance(images, bool):
        raise ProviderNotConfigured("provider option images must be true or false")
    interval = options.get("min_interval_s")
    if interval is not None and (
        isinstance(interval, bool) or not isinstance(interval, int | float) or interval < 0
    ):
        raise ProviderNotConfigured("provider option min_interval_s must be a number of seconds")
    if preset.key_name and api_key(preset.name) is None:
        raise ProviderNotConfigured(f"{preset.key_name} is missing in {env_file()}")
    return ChatCompletionsProvider(preset, images=images, min_interval_s=interval)
