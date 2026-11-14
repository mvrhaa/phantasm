"""Tests for zero.providers.registry."""

import pytest

from zero.config import Config, ProviderConfig
from zero.providers.base import LLMProvider
from zero.providers.chat_completions import ChatCompletionsProvider
from zero.providers.fake import FakeProvider
from zero.providers.registry import ProviderNotConfigured, get_provider, small_model
from zero.secrets import env_file


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


def with_key(line: str) -> None:
    env_file().write_text(line + "\n")


def test_google_with_its_key_in_env() -> None:
    with_key("GEMINI_API_KEY=AIza-test-0123456789")

    provider = get_provider(ProviderConfig(name="google", model="gemma-4-31b-it"))

    assert isinstance(provider, ChatCompletionsProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.name == "google" and provider.supports_images("gemma-4-31b-it")


@pytest.mark.parametrize(
    ("name", "key_name"),
    [
        ("google", "GEMINI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
    ],
)
def test_a_missing_key_leaves_the_provider_not_configured(name: str, key_name: str) -> None:
    with_key("SOME_OTHER_KEY=0123456789")

    with pytest.raises(ProviderNotConfigured) as excinfo:
        get_provider(ProviderConfig(name=name))

    assert str(excinfo.value) == f"{key_name} is missing in {env_file()}"
    assert excinfo.value.code == "provider_not_configured"


def test_a_local_ollama_needs_no_key() -> None:
    assert get_provider(ProviderConfig(name="ollama")).name == "ollama"


def test_options_of_a_preset() -> None:
    with_key("GEMINI_API_KEY=AIza-test-0123456789")

    plain = get_provider(ProviderConfig(name="google", options={"images": False}))
    with pytest.raises(
        ProviderNotConfigured, match="unknown options for the google provider: base_url"
    ):
        get_provider(ProviderConfig(name="google", options={"base_url": "https://evil.example"}))
    with pytest.raises(ProviderNotConfigured, match="images must be true or false"):
        get_provider(ProviderConfig(name="google", options={"images": "yes"}))

    assert not plain.supports_images("gemma-4-31b-it")


def test_the_pause_between_requests() -> None:
    with_key("GEMINI_API_KEY=AIza-test-0123456789")

    google = get_provider(ProviderConfig(name="google"))
    eager = get_provider(ProviderConfig(name="google", options={"min_interval_s": 0}))
    for bad in (-1, "2", True):
        with pytest.raises(ProviderNotConfigured, match="min_interval_s must be a number"):
            get_provider(ProviderConfig(name="google", options={"min_interval_s": bad}))

    assert isinstance(google, ChatCompletionsProvider) and google.min_interval_s == 2.0
    assert isinstance(eager, ChatCompletionsProvider) and eager.min_interval_s == 0
    assert get_provider(ProviderConfig(name="ollama")).min_interval_s == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (ProviderConfig(name="google", model="gemma-4-31b-it"), "gemma-4-26b-a4b-it"),
        (ProviderConfig(name="google", model="x", small_model="tiny"), "tiny"),
        (ProviderConfig(name="openrouter", model="some/model:free"), "some/model:free"),
        (ProviderConfig(), "fake-1"),
    ],
)
def test_the_small_model_comes_from_the_config_the_preset_or_the_main_model(
    config: ProviderConfig, expected: str
) -> None:
    assert small_model(config) == expected
