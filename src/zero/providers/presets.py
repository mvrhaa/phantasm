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
    base_url: str  # without the trailing /chat/completions
    key_name: str | None  # the line of .env holding its key; None for a local server
    images: bool  # whether its models take images, unless the config says otherwise
    stream_usage: bool  # whether to ask for token counts at the end of the stream
    thinking_tokens: int = 0  # added to max_tokens: the model's thinking counts against it
    min_interval_s: float = 0.0  # the least time between two requests, to stay under its limits


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        # Gemma 4 thinks before every answer, and its thinking eats max_tokens.
        Preset(
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "GEMINI_API_KEY",
            images=True,
            stream_usage=True,
            thinking_tokens=4096,
            min_interval_s=2.0,  # free Gemma: about 30 requests a minute
        ),
        Preset(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            images=False,
            stream_usage=True,
            min_interval_s=3.0,  # free models: 20 requests a minute
        ),
        # Mistral sends the token counts on its own and may refuse stream_options.
        Preset(
            "mistral",
            "https://api.mistral.ai/v1",
            "MISTRAL_API_KEY",
            images=True,
            stream_usage=False,
            min_interval_s=1.0,  # free plan: about one request a second
        ),
        Preset("ollama", "http://127.0.0.1:11434/v1", None, images=False, stream_usage=True),
    )
}
