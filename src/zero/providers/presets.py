"""The real providers Zero knows: where each one lives and which key it takes.

All of them speak the ``/chat/completions`` format and share one adapter.
The address and the name of the key are fixed here, in code, so editing the
config can pick a provider but never send its key to another server.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """One provider of the ``/chat/completions`` format."""

    name: str
    base_url: str
    key_name: str | None
    images: bool
    stream_usage: bool
    thinking_tokens: int = 0
    min_interval_s: float = 0.0
    small_model: str | None = None
    echo_call_extra: bool = False


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset(
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "GEMINI_API_KEY",
            images=True,
            stream_usage=True,
            thinking_tokens=4096,
            min_interval_s=2.0,
            small_model="gemma-4-26b-a4b-it",
            echo_call_extra=True,
        ),
        Preset(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            images=False,
            stream_usage=True,
            min_interval_s=3.0,
        ),
        Preset(
            "mistral",
            "https://api.mistral.ai/v1",
            "MISTRAL_API_KEY",
            images=True,
            stream_usage=False,
            min_interval_s=1.0,
        ),
        Preset("ollama", "http://127.0.0.1:11434/v1", None, images=False, stream_usage=True),
    )
}
