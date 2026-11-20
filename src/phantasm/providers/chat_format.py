"""Phantasm's messages in the ``/chat/completions`` format, and the model's answer back.

The request side turns ``Message`` and ``ToolSpec`` into the JSON the API
takes. The answer side puts tool calls together from the pieces a stream
brings and takes the model's thinking out of its text.
"""

import hashlib
import json
import re
import uuid
from typing import Any

from phantasm.protocol.messages import FilePart, ImagePart, TextPart
from phantasm.providers.base import Message, ProviderError, ToolCall, ToolSpec

WIRE_ID = re.compile(r"[A-Za-z0-9]{9}")
TOOL_IMAGES = "Images returned by the tools above:"
THOUGHT_OPEN, THOUGHT_CLOSE = "<thought>", "</thought>"


def wire_id(call_id: str) -> str:
    """A tool call id as the API gets it: 9 letters and digits, which Mistral demands.

    Other ids are hashed, so a call and its result still carry the same one.
    """
    if WIRE_ID.fullmatch(call_id):
        return call_id
    return hashlib.sha256(call_id.encode()).hexdigest()[:9]


def _text(message: Message) -> str:
    pieces = []
    for part in message.content:
        if isinstance(part, TextPart):
            pieces.append(part.text)
        elif isinstance(part, FilePart):
            pieces.append(f"[file {part.name} at {part.path}]")
    return "\n".join(pieces)


def _images(message: Message) -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:{part.mime};base64,{part.data_b64}"}}
        for part in message.content
        if isinstance(part, ImagePart)
    ]


def _with_images(text: str, images: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if not images:
        return text
    return [{"type": "text", "text": text}, *images] if text else images


def _wire_call(call: ToolCall, echo_extra: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": wire_id(call.id),
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.input, ensure_ascii=False)},
    }
    if echo_extra and call.extra:
        entry["extra_content"] = call.extra
    return entry


def _assistant(message: Message, echo_extra: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": _text(message) or None}
    if message.tool_calls:
        entry["tool_calls"] = [_wire_call(call, echo_extra) for call in message.tool_calls]
    elif entry["content"] is None:
        entry["content"] = ""
    return entry


def _tool_result(message: Message) -> dict[str, Any]:
    images = _images(message)
    text = _text(message) or ("(an image, sent in the next message)" if images else "(empty)")
    if message.is_error:
        text = f"Error: {text}"
    return {"role": "tool", "tool_call_id": wire_id(message.tool_call_id or ""), "content": text}


def wire_messages(messages: list[Message], *, echo_extra: bool = False) -> list[dict[str, Any]]:
    """The conversation in the ``/chat/completions`` format.

    A tool result there holds text only, so the images tools returned travel
    in one user message right after the results of that step. ``echo_extra``
    sends each tool call's ``extra`` back, for the provider that attached it.
    """
    wire: list[dict[str, Any]] = []
    held_images: list[dict[str, Any]] = []
    for message in messages:
        if message.role != "tool" and held_images:
            wire.append({"role": "user", "content": _with_images(TOOL_IMAGES, held_images)})
            held_images = []
        if message.role == "tool":
            wire.append(_tool_result(message))
            held_images += _images(message)
        elif message.role == "assistant":
            wire.append(_assistant(message, echo_extra))
        elif message.role == "user":
            wire.append({"role": "user", "content": _with_images(_text(message), _images(message))})
        else:
            wire.append({"role": "system", "content": _text(message)})
    if held_images:
        wire.append({"role": "user", "content": _with_images(TOOL_IMAGES, held_images)})
    return wire


def wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """The tools in the ``/chat/completions`` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


class _CallParts:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""
        self.extra: dict[str, Any] | None = None


class ToolCallAssembler:
    """Puts tool calls together from the pieces a stream brings.

    OpenAI-style streams send the arguments of a call in fragments under one
    ``index``; Mistral sends whole calls, sometimes with the arguments as an
    object; Google sends whole calls with ids of its own.
    """

    def __init__(self) -> None:
        self._calls: list[_CallParts] = []
        self._by_index: dict[Any, _CallParts] = {}

    def add(self, delta: dict[str, Any]) -> None:
        """Take one piece of a call from the ``tool_calls`` of a stream chunk."""
        index = delta.get("index", 0)
        call_id = delta.get("id") or ""
        parts = self._by_index.get(index)
        if parts is None or (call_id and parts.id and call_id != parts.id):
            parts = _CallParts()
            self._calls.append(parts)
            self._by_index[index] = parts
        parts.id = parts.id or call_id
        function = delta.get("function") or {}
        parts.name = parts.name or function.get("name") or ""
        extra = delta.get("extra_content")
        if isinstance(extra, dict):
            parts.extra = {**(parts.extra or {}), **extra}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            parts.arguments += arguments
        elif arguments is not None:
            parts.arguments += json.dumps(arguments)

    def calls(self) -> list[ToolCall]:
        """The finished calls. Broken ones raise a retryable ``ProviderError``."""
        result = []
        for parts in self._calls:
            if not parts.name:
                raise ProviderError("the model sent a tool call without a name", retryable=True)
            try:
                arguments = json.loads(parts.arguments) if parts.arguments.strip() else {}
            except ValueError:
                arguments = None
            if not isinstance(arguments, dict):
                raise ProviderError(
                    f"the model sent broken arguments for {parts.name}", retryable=True
                )
            call_id = parts.id or uuid.uuid4().hex[:9]
            result.append(ToolCall(id=call_id, name=parts.name, input=arguments, extra=parts.extra))
        return result


def _partial_tail(text: str, tag: str) -> str:
    """The end of ``text`` that may be the start of ``tag`` cut by the chunk boundary."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return text[-size:]
    return ""


class ThoughtFilter:
    """Takes the model's thinking out of its text.

    Gemma thinks aloud: ``<thought>…</thought>`` comes before the answer. In
    a stream Google marks those chunks and leaves the closing tag unmarked;
    elsewhere there may be only the tags. A tag cut between two chunks is
    held back until the next chunk tells what it is.
    """

    def __init__(self) -> None:
        self._inside = False
        self._marked = False
        self._held = ""

    def feed(self, text: str, *, thought: bool = False) -> str:
        """The part of the next chunk the user should see."""
        if thought:
            self._inside, self._marked, self._held = True, True, ""
            return ""
        text, self._held = self._held + text, ""
        if self._marked:
            stripped = text.lstrip()
            if len(stripped) < len(THOUGHT_CLOSE) and THOUGHT_CLOSE.startswith(stripped):
                self._held = text
                return ""
            self._inside = self._marked = False
            if stripped.startswith(THOUGHT_CLOSE):
                text = stripped[len(THOUGHT_CLOSE) :]
        visible = []
        while text:
            if self._inside:
                end = text.find(THOUGHT_CLOSE)
                if end < 0:
                    self._held = _partial_tail(text, THOUGHT_CLOSE)
                    break
                text = text[end + len(THOUGHT_CLOSE) :]
                self._inside = False
            else:
                start = text.find(THOUGHT_OPEN)
                if start < 0:
                    self._held = _partial_tail(text, THOUGHT_OPEN)
                    visible.append(text[: len(text) - len(self._held)])
                    break
                visible.append(text[:start])
                text = text[start + len(THOUGHT_OPEN) :]
                self._inside = True
        return "".join(visible)

    def flush(self) -> str:
        """What was held back when the answer ends; nothing if a thought never closed."""
        held, self._held = self._held, ""
        return "" if self._inside else held
