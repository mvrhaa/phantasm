"""Tests of the /chat/completions format: the request, tool calls and thinking."""

import json

import pytest

from phantasm.protocol.messages import FilePart, ImagePart, TextPart
from phantasm.providers.base import Message, ProviderError, ToolCall, ToolSpec
from phantasm.providers.chat_format import (
    TOOL_IMAGES,
    ThoughtFilter,
    ToolCallAssembler,
    wire_id,
    wire_messages,
    wire_tools,
)

PNG = ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")
PNG_URL = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


def test_ids_become_nine_letters_and_digits() -> None:
    google_id = "call_14755307676125612924"

    assert wire_id("abc123XYZ") == "abc123XYZ"
    assert len(wire_id(google_id)) == 9 and wire_id(google_id).isalnum()
    assert wire_id(google_id) == wire_id(google_id) != wire_id("call_1")


def test_a_whole_conversation() -> None:
    call = ToolCall(id="call_1", name="read_file", input={"path": "/tmp/a.txt", "note": "ё"})
    shot = ToolCall(id="abcdefghi", name="screenshot")
    messages = [
        Message(role="system", content=[TextPart(text="You are Phantasm.")]),
        Message(
            role="user",
            content=[
                TextPart(text="Look at this"),
                FilePart(name="a.txt", path="/tmp/a.txt"),
                PNG,
            ],
        ),
        Message(role="assistant", content=[TextPart(text="Reading.")], tool_calls=[call, shot]),
        Message(role="tool", tool_call_id="call_1", content=[TextPart(text="hello")]),
        Message(role="tool", tool_call_id="abcdefghi", content=[PNG]),
        Message(role="assistant", content=[TextPart(text="Done.")]),
        Message(role="user", content=[TextPart(text="Thanks")]),
    ]

    assert wire_messages(messages) == [
        {"role": "system", "content": "You are Phantasm."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this\n[file a.txt at /tmp/a.txt]"},
                PNG_URL,
            ],
        },
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {
                    "id": wire_id("call_1"),
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps(
                            {"path": "/tmp/a.txt", "note": "ё"}, ensure_ascii=False
                        ),
                    },
                },
                {
                    "id": "abcdefghi",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": wire_id("call_1"), "content": "hello"},
        {
            "role": "tool",
            "tool_call_id": "abcdefghi",
            "content": "(an image, sent in the next message)",
        },
        {"role": "user", "content": [{"type": "text", "text": TOOL_IMAGES}, PNG_URL]},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Thanks"},
    ]


def test_errors_empty_results_and_silent_assistants() -> None:
    call = ToolCall(id="abcdefghi", name="run_command", input={})
    messages = [
        Message(role="assistant", tool_calls=[call]),
        Message(role="tool", tool_call_id="abcdefghi", content=[], is_error=True),
        Message(role="assistant"),
    ]

    wire = wire_messages(messages)

    assert wire[0]["content"] is None
    assert wire[1]["content"] == "Error: (empty)"
    assert wire[2] == {"role": "assistant", "content": ""}


def test_images_of_the_last_step_are_not_lost() -> None:
    wire = wire_messages([Message(role="tool", tool_call_id="abcdefghi", content=[PNG])])

    assert wire[-1] == {"role": "user", "content": [{"type": "text", "text": TOOL_IMAGES}, PNG_URL]}


def test_tools() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    assert wire_tools([ToolSpec(name="read_file", description="Read.", input_schema=schema)]) == [
        {
            "type": "function",
            "function": {"name": "read_file", "description": "Read.", "parameters": schema},
        }
    ]


def test_fragments_of_calls_are_put_together() -> None:
    calls = ToolCallAssembler()
    calls.add({"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"pa'}})
    calls.add({"index": 1, "id": "c2", "function": {"name": "list_dir", "arguments": ""}})
    calls.add({"index": 0, "function": {"arguments": 'th": "/tmp"}'}})

    assert calls.calls() == [
        ToolCall(id="c1", name="read_file", input={"path": "/tmp"}),
        ToolCall(id="c2", name="list_dir", input={}),
    ]


def test_whole_calls_with_object_arguments_and_a_reused_index() -> None:
    calls = ToolCallAssembler()
    calls.add({"id": "aaaaaaaaa", "function": {"name": "a", "arguments": {"x": 1}}})
    calls.add({"id": "bbbbbbbbb", "function": {"name": "b", "arguments": {"y": 2}}})
    calls.add({"function": {"name": "c", "arguments": "{}"}, "index": 5})

    first, second, third = calls.calls()

    assert (first.name, first.input, second.name, second.input) == ("a", {"x": 1}, "b", {"y": 2})
    assert len(third.id) == 9


@pytest.mark.parametrize(
    "function",
    [{"arguments": "{}"}, {"name": "a", "arguments": "{broken"}, {"name": "a", "arguments": "[1]"}],
)
def test_broken_calls_are_worth_another_try(function: dict[str, str]) -> None:
    calls = ToolCallAssembler()
    calls.add({"id": "c1", "function": function})

    with pytest.raises(ProviderError) as excinfo:
        calls.calls()
    assert excinfo.value.retryable


def visible(*pieces: str | tuple[str, str]) -> str:
    """Run text through a ThoughtFilter; ("thought", text) is a chunk marked as thinking."""
    thoughts = ThoughtFilter()
    out = []
    for piece in pieces:
        if isinstance(piece, tuple):
            out.append(thoughts.feed(piece[1], thought=True))
        else:
            out.append(thoughts.feed(piece))
    return "".join(out) + thoughts.flush()


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (("<thought>plan</thought>Answer",), "Answer"),
        (("<tho", "ught>plan</th", "ought>Ans", "wer"), "Answer"),
        (("Hello <", "b>world"), "Hello <b>world"),
        (("Text ends with <tho",), "Text ends with <tho"),
        (("<thought>never closed",), ""),
        (("A<thought>x</thought>B<thought>y</thought>C",), "ABC"),
        ((("thought", "<thought>plan"), ("thought", " more"), "</thought>", "Answer"), "Answer"),
        ((("thought", "plan"), "Answer"), "Answer"),
        ((("thought", "plan"), "</tho", "ught>Hi"), "Hi"),
        ((("thought", "plan"), " ", "\n</thought>Hi"), "Hi"),
        (("Plain text",), "Plain text"),
    ],
)
def test_thinking_is_taken_out(pieces: tuple[str | tuple[str, str], ...], expected: str) -> None:
    assert visible(*pieces) == expected
