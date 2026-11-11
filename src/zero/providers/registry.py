"""Choosing the provider named in the config."""

from zero.config import ProviderConfig
from zero.protocol.errors import ErrorCode
from zero.providers.base import LLMProvider
from zero.providers.fake import FakeProvider


class ProviderNotConfigured(Exception):
    """The config names a provider that does not exist or gives it bad options."""

    code = ErrorCode.PROVIDER_NOT_CONFIGURED


def get_provider(config: ProviderConfig) -> LLMProvider:
    """Build the provider for ``config.name``, passing ``config.options`` to it."""
    if config.name == "fake":
        try:
            return FakeProvider(**config.options)
        except TypeError as exc:
            raise ProviderNotConfigured(f"bad options for the fake provider: {exc}") from exc
    raise ProviderNotConfigured(f"unknown provider {config.name!r}")
