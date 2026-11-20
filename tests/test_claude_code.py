"""The Claude Code provider, run against ``tests/fake_claude.py`` (no network, no real Claude)."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from phantasm.core.cancellation import CancelToken
from phantasm.protocol.messages import ImagePart, TextPart
from phantasm.providers import claude_format as fmt
from phantasm.providers.base import (
    Message,
    ProviderError,
    StopEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)
from phantasm.providers.claude_code import ClaudeCodeProvider

FAKE = Path(__file__).with_name("fake_claude.py")
NOTE = ToolSpec(
    name="read_note",
    description="Read a note.",
    input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
)
PIXEL = ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "fake-claude"
    folder.mkdir()
    (folder / "turns.json").write_text("[]")
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(folder))
    return folder


@pytest.fixture
async def provider(tmp_path: Path, fake: Path):
    made = ClaudeCodeProvider(tmp_path / "claude-code", command=[sys.executable, str(FAKE)])
    yield made
    await made.aclose()


def turns(fake: Path, *planned: list[dict[str, Any]]) -> None:
    (fake / "turns.json").write_text(json.dumps(list(planned)))


def log(fake: Path) -> list[dict[str, Any]]:
    path = fake / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def starts(fake: Path) -> list[list[str]]:
    return [entry["argv"] for entry in log(fake) if "argv" in entry]


def user_lines(fake: Path) -> list[list[dict[str, Any]]]:
    return [
        entry["in"]["message"]["content"]
        for entry in log(fake)
        if "in" in entry and entry["in"].get("type") == "user"
    ]


def system(text: str = "You are Phantasm.") -> Message:
    return Message(role="system", content=[TextPart(text=text)])


def user(text: str, *extra: Any) -> Message:
    return Message(role="user", content=[TextPart(text=text), *extra])


async def collect(provider: ClaudeCodeProvider, messages, tools=(), cancel=None):
    events = []
    async for event in provider.stream(
        messages, list(tools), "sonnet", cancel=cancel or CancelToken()
    ):
        events.append(event)
    return events


def text(events) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


async def test_plain_answer_streams_text_usage_and_end(provider, fake):
    turns(fake, [{"text": "Hello there"}])
    events = await collect(provider, [system(), user("hi")])
    assert text(events) == "Hello there "
    usage = next(event for event in events if isinstance(event, UsageEvent))
    assert (usage.input_tokens, usage.output_tokens) == (15, 7)
    assert events[-1] == StopEvent(reason="end")
    argv = starts(fake)[0]
    assert "--no-session-persistence" in argv and "--mcp-config" not in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert user_lines(fake) == [[{"type": "text", "text": "hi"}]]


async def test_tool_call_goes_on_in_the_same_process(provider, fake):
    turns(
        fake,
        [
            {"text": "Let me look", "tool": {"name": "read_note", "input": {"name": "a"}}},
            {"text": "It says milk"},
        ],
    )
    prompt = [system(), user("what is in note a?")]
    first = await collect(provider, prompt, [NOTE])
    calls = [event.call for event in first if isinstance(event, ToolCallEvent)]
    assert [(call.name, call.input) for call in calls] == [("read_note", {"name": "a"})]
    assert first[-1] == StopEvent(reason="tool_use")
    prompt += [
        Message(role="assistant", content=[TextPart(text="Let me look ")], tool_calls=calls),
        Message(role="tool", content=[TextPart(text="milk"), PIXEL], tool_call_id=calls[0].id),
    ]
    second = await collect(provider, prompt, [NOTE])
    assert text(second) == "It says milk "
    assert len(starts(fake)) == 1
    assert {
        "name": "read_note",
        "description": "Read a note.",
        "inputSchema": NOTE.input_schema,
    } in next(e["tools"] for e in log(fake) if "tools" in e)
    result = next(entry["result"] for entry in log(fake) if "result" in entry)
    assert result == {"content": [{"type": "text", "text": "milk"}], "isError": False}
    images = user_lines(fake)[1]
    assert images[0] == {"type": "text", "text": fmt.TOOL_IMAGES}
    assert images[1]["source"]["data"] == PIXEL.data_b64


async def test_next_turn_goes_on_without_a_new_process(provider, fake):
    turns(fake, [{"text": "One"}], [{"text": "Two"}])
    prompt = [system("time 10:00"), user("first")]
    await collect(provider, prompt, [NOTE])
    prompt = [
        system("time 10:01"),
        user("first"),
        Message(role="assistant", content=[TextPart(text="One ")]),
        user("second"),
    ]
    events = await collect(provider, prompt, [NOTE])
    assert text(events) == "Two "
    assert len(starts(fake)) == 1
    assert user_lines(fake)[-1] == [{"type": "text", "text": "second"}]


async def test_closed_conversation_resumes_its_session(provider, fake):
    turns(fake, [{"text": "One"}], [{"text": "Two"}])
    await collect(provider, [system(), user("first")], [NOTE])
    await provider.aclose()
    prompt = [
        system(),
        user("first"),
        Message(role="assistant", content=[TextPart(text="One")]),
        user("second"),
    ]
    await collect(provider, prompt, [NOTE])
    first, second = starts(fake)
    assert "--resume" not in first
    assert second[second.index("--resume") + 1]
    assert user_lines(fake)[-1] == [{"type": "text", "text": "second"}]


async def test_gone_session_starts_over_with_a_record(provider, fake):
    turns(fake, [{"text": "One"}], [{"text": "Two"}])
    await collect(provider, [system(), user("first")], [NOTE])
    await provider.aclose()
    for session in (fake / "sessions").iterdir():
        session.unlink()
    prompt = [
        system(),
        user("first"),
        Message(role="assistant", content=[TextPart(text="One")]),
        user("second"),
    ]
    events = await collect(provider, prompt, [NOTE])
    assert text(events) == "Two "
    assert "--resume" not in starts(fake)[-1]
    record, new = user_lines(fake)[-1]
    assert record["text"].startswith(fmt.RECORD_INTRO)
    assert "User: first" in record["text"] and "You: One" in record["text"]
    assert new == {"type": "text", "text": "second"}


async def test_unknown_history_starts_a_session_with_a_record(provider, fake):
    turns(fake, [{"text": "Sure"}])
    call = ToolCall(id="c1", name="read_note", input={"name": "a"})
    prompt = [
        system(),
        user("old"),
        Message(role="assistant", content=[], tool_calls=[call]),
        Message(role="tool", content=[TextPart(text="milk")], tool_call_id="c1"),
        Message(role="assistant", content=[TextPart(text="It is milk")]),
        user("thanks", PIXEL),
    ]
    await collect(provider, prompt, [NOTE])
    record, new, image = user_lines(fake)[0]
    assert 'You called the tool read_note with {"name": "a"}' in record["text"]
    assert "The tool read_note returned: milk" in record["text"]
    assert new == {"type": "text", "text": "thanks"} and image["type"] == "image"


async def test_cancel_ends_quietly_and_drops_the_conversation(provider, fake):
    turns(
        fake,
        [{"text": "Let me look", "tool": {"name": "read_note", "input": {}}}],
        [{"text": "fresh"}],
    )
    prompt = [system(), user("look")]
    first = await collect(provider, prompt, [NOTE])
    calls = [event.call for event in first if isinstance(event, ToolCallEvent)]
    prompt += [
        Message(role="assistant", content=[], tool_calls=calls),
        Message(role="tool", content=[TextPart(text="x")], tool_call_id=calls[0].id),
    ]
    cancel = CancelToken()
    cancel.cancel()
    assert await collect(provider, prompt, [NOTE], cancel=cancel) == []
    await collect(provider, [system(), user("again")], [NOTE])
    assert len(starts(fake)) == 2


async def test_failed_turn_raises_its_message(provider, fake):
    turns(fake, [{"error": "Not logged in · Please run /login"}])
    with pytest.raises(ProviderError, match="Please run /login"):
        await collect(provider, [system(), user("hi")])


async def test_crash_raises_its_last_error_line(provider, fake):
    turns(fake, [{"crash": "error: unknown option '--safe-mode'"}])
    with pytest.raises(ProviderError, match="unknown option"):
        await collect(provider, [system(), user("hi")])


async def test_missing_command_is_a_provider_error(tmp_path, fake):
    missing = ClaudeCodeProvider(tmp_path / "cc", command=[str(tmp_path / "no-such-claude")])
    with pytest.raises(ProviderError, match="is it installed"):
        await collect(missing, [system(), user("hi")])


async def test_last_step_without_tools_asks_for_an_answer(provider, fake):
    turns(
        fake,
        [{"text": "", "tool": {"name": "read_note", "input": {}}}, {"text": "Here is what I have"}],
    )
    prompt = [system(), user("look")]
    first = await collect(provider, prompt, [NOTE])
    calls = [event.call for event in first if isinstance(event, ToolCallEvent)]
    prompt += [
        Message(role="assistant", content=[], tool_calls=calls),
        Message(role="tool", content=[TextPart(text="x")], tool_call_id=calls[0].id),
    ]
    events = await collect(provider, prompt, [])
    assert text(events) == "Here is what I have "
    assert user_lines(fake)[-1] == [{"type": "text", "text": fmt.NO_TOOLS}]
    assert len(starts(fake)) == 1


async def test_tools_change_moves_an_ended_conversation_to_a_new_process(provider, fake):
    turns(fake, [{"text": "One"}], [{"text": "Two"}])
    await collect(provider, [system(), user("first")], [NOTE])
    prompt = [
        system(),
        user("first"),
        Message(role="assistant", content=[TextPart(text="One")]),
        user("again"),
    ]
    await collect(provider, prompt, [])
    first, second = starts(fake)
    assert "--mcp-config" in first and "--mcp-config" not in second
    assert "--resume" not in second
    assert user_lines(fake)[-1][0]["text"].startswith(fmt.RECORD_INTRO)


def test_key_ignores_images_and_their_placeholders():
    plain = user("look")
    assert fmt.key(plain) == fmt.key(user("look", PIXEL))
    assert fmt.key(plain) == fmt.key(user("look", TextPart(text="[image omitted]")))
    assert fmt.key(plain) != fmt.key(user("look!"))


def test_new_input_is_the_trailing_user_messages():
    reply = Message(role="assistant", content=[TextPart(text="a")])
    assert fmt.new_input([user("a"), reply, user("b"), user("c")]) == 2
    assert fmt.new_input([user("a")]) == 0
    tool = Message(role="tool", content=[], tool_call_id="x")
    assert fmt.first_message([user("a"), tool])[-1]["text"] == fmt.GO_ON


async def test_idle_processes_are_limited(provider, fake, monkeypatch):
    monkeypatch.setattr("phantasm.providers.claude_code.MAX_IDLE", 1)
    turns(fake, [{"text": "One"}], [{"text": "Two"}])
    await collect(provider, [system(), user("first")], [NOTE])
    await collect(provider, [system(), user("other")], [NOTE])
    await asyncio.sleep(0.5)
    alive = [c for c in provider._conversations if c.alive]
    assert len(alive) == 1
