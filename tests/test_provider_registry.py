"""Tests for zero.providers.registry."""

import pytest

from zero.config import Config, ProviderConfig
from zero.providers.base import LLMProvider
from zero.providers.fake import FakeProvider
from zero.providers.registry import ProviderNotConfigured, get_provider


def test_default_config_gives_the_fake_provider_in_echo_mode() -> None:
    provider = get_provider(Config().provider)

    assert isinstance(provider, FakeProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.infinite is False
    assert provider.delay_s == 0


def test_options_are_passed_to_the_fake_provider() -> None:
    config = ProviderConfig(
        options={"delay_s": 0.1, "fail_times": 2, "fail_retryable": False, "infinite": True}
    )

    provider = get_provider(config)

    assert isinstance(provider, FakeProvider)
    assert (provider.delay_s, provider.fail_retryable, provider.infinite) == (0.1, False, True)


def test_each_call_builds_a_new_provider() -> None:
    config = Config().provider

    assert get_provider(config) is not get_provider(config)


@pytest.mark.parametrize("name", ["openai", "anthropic", "", "Fake"])
def test_unknown_provider_is_not_configured(name: str) -> None:
    with pytest.raises(ProviderNotConfigured, match="unknown provider") as excinfo:
        get_provider(ProviderConfig(name=name))

    assert excinfo.value.code == "provider_not_configured"


def test_unknown_option_is_not_configured() -> None:
    with pytest.raises(ProviderNotConfigured, match="bad options"):
        get_provider(ProviderConfig(options={"temperature": 0.2}))
