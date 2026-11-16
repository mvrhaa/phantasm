"""Zero's messages as Claude Code takes them, and how to tell that a conversation goes on.

Claude Code keeps a conversation of its own, so the provider compares what the
loop sends with what it already told Claude: messages are compared by what they
say, images left out (the loop shrinks old ones to placeholders).
"""

import hashlib
import json
from typing import Any

from zero.protocol.messages import FilePart, ImagePart, TextPart
from zero.providers.base import Message, ToolSpec

PLACEHOLDERS = frozenset({"[image omitted]", "[an image; this model cannot see images]"})
TOOL_IMAGES = "Images returned by the tools above:"
RECORD_INTRO = (
    "[This conversation goes on in a new session, so what came before is here as a record, "
    "oldest first: the user's messages, your replies, your tool calls and their results. "
    "Go on from its end as yourself and call tools as you always do; the notation below "
    "is only a record.]"
)
RECORD_END = "[The record ends here.]"
GO_ON = "[Your tool calls' results are in the record: go on.]"
NO_TOOLS = "[No more tools in this turn: answer now with what you have.]"


def digest(value: Any) -> str:
    """A short stable hash of any JSON value."""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_image_file(part: FilePart) -> bool:
    return (part.mime or "").startswith("image/")


def text_of(message: Message) -> str:
    """The words of a message: its text and the files it names, no images."""
    pieces = []
    for part in message.content:
        if isinstance(part, TextPart) and part.text not in PLACEHOLDERS:
            pieces.append(part.text)
        elif isinstance(part, FilePart) and not _is_image_file(part):
            pieces.append(f"[file {part.name} at {part.path}]")
    return "\n".join(pieces)


def key(message: Message) -> str:
    """What a message says, to compare it with what Claude was told."""
    calls = [[call.id, call.name, call.input] for call in message.tool_calls]
    return digest([message.role, text_of(message), calls, message.tool_call_id])


def images(message: Message) -> list[dict[str, Any]]:
    """The message's images as Claude's image blocks."""
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": part.mime, "data": part.data_b64},
        }
        for part in message.content
        if isinstance(part, ImagePart)
    ]


def blocks(message: Message) -> list[dict[str, Any]]:
    """A user message as Claude's content blocks: its text, then its images."""
    found: list[dict[str, Any]] = []
    text = text_of(message)
    if text:
        found.append({"type": "text", "text": text})
    return found + images(message)


def mcp_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Zero's tools as an MCP server lists them."""
    return [
        {"name": spec.name, "description": spec.description, "inputSchema": spec.input_schema}
        for spec in specs
    ]


def tool_result(message: Message) -> dict[str, Any]:
    """A tool message as the MCP result of its call (its images go in a user message)."""
    return {
        "content": [{"type": "text", "text": text_of(message) or "(no output)"}],
        "isError": message.is_error,
    }


def new_input(messages: list[Message]) -> int:
    """Where the trailing user messages start: what is new for Claude in a fresh session."""
    start = len(messages)
    while start > 0 and messages[start - 1].role == "user":
        start -= 1
    return start


def record(messages: list[Message]) -> str:
    """Earlier messages as a plain record for a new session."""
    lines = [RECORD_INTRO]
    names: dict[str, str] = {}
    for message in messages:
        text = text_of(message)
        if message.role == "user":
            lines.append(f"User: {text}" if text else "User: (an image)")
        elif message.role == "assistant":
            if text:
                lines.append(f"You: {text}")
            for call in message.tool_calls:
                names[call.id] = call.name
                arguments = json.dumps(call.input, ensure_ascii=False)
                lines.append(f"You called the tool {call.name} with {arguments}")
        elif message.role == "tool":
            name = names.get(message.tool_call_id or "", "a tool")
            status = "failed" if message.is_error else "returned"
            lines.append(f"The tool {name} {status}: {text or '(no output)'}")
    lines.append(RECORD_END)
    return "\n\n".join(lines)


def first_message(messages: list[Message]) -> list[dict[str, Any]]:
    """The first message of a new session: earlier messages as a record, then the new ones."""
    start = new_input(messages)
    found: list[dict[str, Any]] = []
    if start > 0:
        found.append({"type": "text", "text": record(messages[:start])})
    for message in messages[start:]:
        found.extend(blocks(message))
    if start == len(messages):
        found.append({"type": "text", "text": GO_ON})
    return found
