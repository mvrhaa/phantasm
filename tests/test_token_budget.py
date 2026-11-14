"""Tests of the token estimate and of the history budget a model gets."""

import pytest
from pydantic import ValidationError

from tests.turns import Harness
from zero.config import Config, ProviderConfig
from zero.core.context import estimate_tokens, text_tokens
from zero.core.prompting import PromptBuilder
from zero.protocol.messages import TextPart
from zero.providers.base import Message, ToolCall
from zero.providers.fake import FakeProvider


def test_cyrillic_counts_for_more_than_english() -> None:
    assert text_tokens("a" * 35) == pytest.approx(10)
    assert text_tokens("я" * 25) == pytest.approx(10)
    assert text_tokens("ab" + "яя") == pytest.approx(2 / 3.5 + 2 / 2.5)
    assert text_tokens("") == 0


def test_a_russian_message_is_estimated_larger_than_an_english_one() -> None:
    def message(text: str) -> Message:
        return Message(role="user", content=[TextPart(text=text)])

    english = estimate_tokens(message("hello " * 100))
    russian = estimate_tokens(message("привет " * 100))

    assert english == 172
    assert russian == 269


def test_a_thought_signature_does_not_count() -> None:
    call = ToolCall(id="c1", name="read_file", input={"path": "~/notes.md"})
    signed = call.model_copy(update={"extra": {"google": {"thought_signature": "x" * 4000}}})

    plain_size = estimate_tokens(Message(role="assistant", tool_calls=[call]))
    signed_size = estimate_tokens(Message(role="assistant", tool_calls=[signed]))

    assert signed_size == plain_size


def test_tool_calls_count_by_the_same_rule() -> None:
    call = ToolCall(id="c", name="write_file", input={"content": "ж" * 50})
    message = Message(role="assistant", content=[], tool_calls=[call])

    dumped = call.model_dump_json(exclude={"extra"})
    assert estimate_tokens(message) == pytest.approx(text_tokens(call.name + dumped), abs=1)
    assert estimate_tokens(message) > len(call.name + dumped) / 3.5


async def stored_turns(harness: Harness, count: int, chars: int) -> None:
    for n in range(count):
        await harness.messages.append(
            harness.session_id, "user", [TextPart(text=f"q{n} " + "x" * chars)], status="ok"
        )
        await harness.messages.append(
            harness.session_id, "assistant", [TextPart(text=f"a{n}")], status="ok"
        )


def builder(harness: Harness, config: Config) -> PromptBuilder:
    return PromptBuilder(
        session_id=harness.session_id,
        client=None,
        source="desktop",
        config=config,
        provider=FakeProvider(),
        messages=harness.messages,
        events=harness.bus,
    )


async def test_the_models_own_budget_wins_over_the_default(harness: Harness) -> None:
    await stored_turns(harness, count=6, chars=1400)
    roomy = Config()
    tight = Config()
    tight.provider.history_token_budget = 1000

    everything = await builder(harness, roomy).build()
    latest = await builder(harness, tight).build()

    assert len(everything) == 1 + 12
    assert len(latest) == 1 + 4
    assert latest[-2].content[0].text.startswith("q5 ")  # type: ignore[union-attr]


def test_a_models_budget_must_be_sensible() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(history_token_budget=10)
    assert ProviderConfig().history_token_budget is None
