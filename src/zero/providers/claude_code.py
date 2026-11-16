"""Thinking through the user's own Claude Code (config ``claude_code``).

The turn loop stays Zero's: it builds the prompt, runs the tools, asks for
approvals. Claude Code holds a conversation of its own, so one process is kept
for each conversation the loop has, and the loop's next request goes on in it
when it continues what that process was told. Zero's tools reach Claude through
an MCP server the provider serves over Claude Code's control channel: when
Claude calls one, ``stream`` returns the call; the next ``stream`` brings its
result. Claude Code sees only Zero's prompt and tools: none of the user's
settings, CLAUDE.md, plugins, MCP servers or built-in tools.

A conversation that ended its turn closes after ``IDLE_S`` and goes on later
in a new process (``--resume``); so does one whose prompt is older than
``FRESH_S`` (the prompt holds the time). When nothing continues what the loop
sends, a new session starts with the earlier messages as a record.
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from zero.providers import claude_format as fmt
from zero.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)
from zero.providers.claude_process import ClaudeProcess, child_env

logger = logging.getLogger(__name__)

NAME = "claude_code"
SERVER = "zero"
PREFIX = f"mcp__{SERVER}__"
BASE_ARGS = [
    "-p",
    "--input-format", "stream-json",
    "--output-format", "stream-json",
    "--verbose",
    "--include-partial-messages",
    "--tools", "",
    "--restricted",
    "--safe-mode",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--system-prompt-snapshot", "off",
]  # fmt: skip
TOOL_ARGS = [
    "--mcp-config", json.dumps({"mcpServers": {SERVER: {"type": "sdk", "name": SERVER}}}),
    "--permission-mode", "manual",
    "--permission-prompts", "host",
    "--permission-prompt-tool", "stdio",
]  # fmt: skip
ENV = {"ENABLE_TOOL_SEARCH": "false", "MCP_TOOL_TIMEOUT": str(7 * 24 * 3600 * 1000)}
IDLE_S = 600.0
WAITING_S = 3600.0
FRESH_S = 1800.0
MAX_IDLE = 2
MAX_RECORDS = 50
GONE_MARK = "No conversation found"


class _Gone(Exception):
    """The session a conversation was to resume is gone (Claude Code deletes old ones)."""


@dataclass
class Conversation:
    """One Claude Code session, its process if one runs, and what it was told."""

    model: str
    tools_key: str
    prompt_key: str
    seen: list[str] = field(default_factory=list)
    session: str = ""
    process: ClaudeProcess | None = None
    prompt_file: Path | None = None
    started: float = 0.0
    used: float = 0.0
    tools: list[dict[str, Any]] = field(default_factory=list)
    calls: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    waiting: list[str] = field(default_factory=list)
    ended: bool = True
    busy: bool = False
    persist: bool = False
    timer: asyncio.TimerHandle | None = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.alive


class ClaudeCodeProvider:
    """An ``LLMProvider`` that thinks with Claude through the ``claude`` command."""

    name = NAME
    writes_prose = True

    def __init__(self, folder: Path, command: Sequence[str] = ("claude",)) -> None:
        self._folder = folder
        self._command = list(command)
        self._conversations: list[Conversation] = []
        for stale in folder.glob("prompt-*.md") if folder.is_dir() else []:
            stale.unlink(missing_ok=True)

    def supports_images(self, model: str) -> bool:
        return True

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        return None

    async def aclose(self) -> None:
        """End every process (they save their sessions)."""
        await asyncio.gather(*(self._close(c) for c in self._conversations if c.alive))

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        system = fmt.text_of(messages[0]) if messages and messages[0].role == "system" else ""
        rest = messages[1:] if messages and messages[0].role == "system" else list(messages)
        keys = [fmt.key(message) for message in rest]
        tools_key = fmt.digest([spec.model_dump() for spec in tools])
        for attempt in range(2):
            found = None if attempt else self._find(rest, keys, model, tools_key)
            if found is None:
                conversation = Conversation(model, tools_key, fmt.digest(system))
                self._conversations.append(conversation)
                tail: list[Message] = []
            else:
                conversation, tail = found
            conversation.busy = True
            if conversation.timer is not None:
                conversation.timer.cancel()
            clean = False
            try:
                await self._send(conversation, system, tools, rest, tail, fresh=found is None)
                async for event in self._read(conversation, cancel, final=not tools):
                    yield event
                conversation.seen = keys
                clean = True
                return
            except _Gone:
                clean = True
                self._forget(conversation)
                continue
            except _Cancelled:
                return
            finally:
                conversation.busy = False
                conversation.used = time.monotonic()
                if clean:
                    self._after(conversation)
                else:
                    self._drop(conversation)
        raise ProviderError("Claude Code could not resume the conversation")

    # ---------- Which conversation goes on ----------

    def _find(
        self, rest: list[Message], keys: list[str], model: str, tools_key: str
    ) -> tuple[Conversation, list[Message]] | None:
        for conversation in sorted(self._conversations, key=lambda c: -c.used):
            if conversation.busy or conversation.model != model:
                continue
            if conversation.waiting and not conversation.alive:
                continue
            count = len(conversation.seen)
            if not count or keys[:count] != conversation.seen:
                continue
            tail = rest[count:]
            if not self._continues(conversation, tail):
                continue
            same_tools = conversation.tools_key == tools_key
            if conversation.waiting and not same_tools and tools_key != fmt.digest([]):
                continue
            return conversation, tail
        return None

    @staticmethod
    def _continues(conversation: Conversation, tail: list[Message]) -> bool:
        """Whether ``tail`` goes on from where the conversation is: its own reply first,
        then the results of the calls it waits for, then the user's words."""
        index = 0
        if tail and tail[0].role == "assistant":
            if [call.id for call in tail[0].tool_calls] != conversation.waiting:
                return False
            index = 1
        elif conversation.waiting:
            return False
        results = tail[index : index + len(conversation.waiting)]
        if conversation.waiting:
            ids = [message.tool_call_id for message in results if message.role == "tool"]
            if sorted(ids, key=str) != sorted(conversation.waiting):
                return False
            index += len(conversation.waiting)
        said = tail[index:]
        if any(message.role != "user" for message in said):
            return False
        return bool(said) or bool(conversation.waiting)

    # ---------- Processes ----------

    async def _start(self, conversation: Conversation, system: str, tools: list[ToolSpec]) -> None:
        self._folder.mkdir(parents=True, exist_ok=True)
        prompt = self._folder / f"prompt-{uuid.uuid4().hex}.md"
        prompt.write_text(system, encoding="utf-8")
        args = [*BASE_ARGS, "--system-prompt-file", str(prompt), "--model", conversation.model]
        if tools:
            args += TOOL_ARGS
        else:
            args += ["--no-session-persistence"]
        if conversation.session and tools:
            args += ["--resume", conversation.session]
        else:
            conversation.session = ""
        conversation.persist = bool(tools)
        conversation.tools = fmt.mcp_tools(tools)
        conversation.tools_key = fmt.digest([spec.model_dump() for spec in tools])
        conversation.prompt_key = fmt.digest(system)
        conversation.prompt_file = prompt
        conversation.calls.clear()
        conversation.results.clear()
        conversation.waiting = []
        conversation.ended = True
        try:
            conversation.process = await ClaudeProcess.start(
                [*self._command, *args],
                cwd=self._folder,
                env=child_env(ENV),
                on_control=partial(self._control, conversation),
            )
        except OSError as exc:
            prompt.unlink(missing_ok=True)
            raise ProviderError(
                f"Claude Code could not start ({type(exc).__name__}); is it installed?"
            ) from None
        conversation.started = time.monotonic()
        self._limit_idle(conversation)

    async def _send(
        self,
        conversation: Conversation,
        system: str,
        tools: list[ToolSpec],
        rest: list[Message],
        tail: list[Message],
        *,
        fresh: bool,
    ) -> None:
        """Tell the conversation what is new, in a new process if it needs one.

        A conversation that ended its turn moves to a new process when its tools
        changed, or its prompt changed and is old. It resumes its session there;
        without one (tool-less conversations keep none on disk) it starts over
        with the earlier messages as a record.
        """
        tools_key = fmt.digest([spec.model_dump() for spec in tools])
        stale = time.monotonic() - conversation.started > FRESH_S
        changed = conversation.tools_key != tools_key or (
            stale and conversation.prompt_key != fmt.digest(system)
        )
        if fresh or not conversation.alive or (conversation.ended and changed):
            if conversation.alive:
                await self._close(conversation)
            await self._start(conversation, system, tools)
            if not conversation.session:
                await self._say(conversation, fmt.first_message(rest))
                return
        await self._feed(conversation, tools, tail)

    async def _say(self, conversation: Conversation, content: list[dict[str, Any]]) -> None:
        if conversation.process is not None:
            await conversation.process.say(content)
        conversation.ended = False

    async def _feed(
        self, conversation: Conversation, tools: list[ToolSpec], tail: list[Message]
    ) -> None:
        """The results of the calls it waits for, and the user's words after them (sent
        first: Claude Code gives them to Claude right after the results)."""
        said: list[dict[str, Any]] = []
        pictures = [block for m in tail if m.role == "tool" for block in fmt.images(m)]
        if pictures:
            said += [{"type": "text", "text": fmt.TOOL_IMAGES}, *pictures]
        for message in tail:
            if message.role == "user":
                said += fmt.blocks(message)
        if conversation.waiting and not tools and conversation.tools:
            said.append({"type": "text", "text": fmt.NO_TOOLS})
        if said:
            await self._say(conversation, said)
        for message in tail:
            if message.role == "tool" and message.tool_call_id in conversation.waiting:
                await self._give(conversation, message.tool_call_id, fmt.tool_result(message))
        conversation.waiting = []
        conversation.ended = False

    async def _close(self, conversation: Conversation) -> None:
        process, conversation.process = conversation.process, None
        if process is not None:
            await process.close()
        self._cleanup(conversation)

    def _cleanup(self, conversation: Conversation) -> None:
        if conversation.prompt_file is not None:
            conversation.prompt_file.unlink(missing_ok=True)
            conversation.prompt_file = None
        if conversation.timer is not None:
            conversation.timer.cancel()
            conversation.timer = None

    def _forget(self, conversation: Conversation) -> None:
        with contextlib.suppress(ValueError):
            self._conversations.remove(conversation)
        self._cleanup(conversation)

    def _drop(self, conversation: Conversation) -> None:
        """A conversation that broke off mid-answer: its process and session are not used again."""
        if conversation.process is not None:
            conversation.process.kill()
        conversation.process = None
        self._forget(conversation)

    def _after(self, conversation: Conversation) -> None:
        """Close the process later: soon at the end of a turn, after a long while when it waits."""
        if conversation not in self._conversations:
            return
        delay = WAITING_S if conversation.waiting else IDLE_S
        loop = asyncio.get_running_loop()
        conversation.timer = loop.call_later(delay, self._expire, conversation)
        records = [c for c in self._conversations if not c.alive]
        for old in sorted(records, key=lambda c: c.used)[: max(len(records) - MAX_RECORDS, 0)]:
            self._forget(old)

    def _expire(self, conversation: Conversation) -> None:
        conversation.timer = None
        if conversation.busy:
            return
        if conversation.waiting:
            self._drop(conversation)
        else:
            asyncio.get_running_loop().create_task(self._close(conversation))

    def _limit_idle(self, keep: Conversation) -> None:
        """Keep at most ``MAX_IDLE`` processes that only wait for the next turn."""
        idle = [
            c
            for c in self._conversations
            if c is not keep and c.alive and c.ended and not c.busy and not c.waiting
        ]
        for old in sorted(idle, key=lambda c: c.used)[: max(len(idle) - MAX_IDLE + 1, 0)]:
            asyncio.get_running_loop().create_task(self._close(old))

    # ---------- What Claude Code asks ----------

    async def _control(
        self, conversation: Conversation, process: ClaudeProcess, event: dict[str, Any]
    ) -> None:
        request = event.get("request") or {}
        request_id = event.get("request_id")
        subtype = request.get("subtype")
        if subtype == "can_use_tool":
            allow = {"behavior": "allow", "updatedInput": request.get("input") or {}}
            await process.answer(request_id, allow)
        elif subtype == "mcp_message":
            await self._mcp(conversation, process, request_id, request.get("message") or {})
        else:
            await process.answer(request_id, error=f"Zero does not answer {subtype}")

    async def _mcp(
        self,
        conversation: Conversation,
        process: ClaudeProcess,
        request_id: Any,
        message: dict[str, Any],
    ) -> None:
        method, mid = message.get("method"), message.get("id")
        params = message.get("params") or {}
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": mid}
        if method == "initialize":
            reply["result"] = {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER, "version": "1"},
            }
        elif method == "tools/list":
            reply["result"] = {"tools": conversation.tools}
        elif method == "tools/call":
            use = (params.get("_meta") or {}).get("claudecode/toolUseId", "")
            result = conversation.results.pop(use, None)
            if result is None:
                conversation.calls[use] = (request_id, mid)
                return
            reply["result"] = result
        elif mid is None or method == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {"code": -32601, "message": f"Zero does not serve {method}"}
        await process.answer(request_id, {"mcp_response": reply})

    async def _give(self, conversation: Conversation, use: str, result: dict[str, Any]) -> None:
        waiting = conversation.calls.pop(use, None)
        if waiting is None:
            conversation.results[use] = result
            return
        process = conversation.process
        if process is not None:
            response = {"jsonrpc": "2.0", "id": waiting[1], "result": result}
            await process.answer(waiting[0], {"mcp_response": response})

    # ---------- Reading the answer ----------

    async def _next(self, process: ClaudeProcess, cancel: CancelSignal) -> Any:
        """The next event, ``None`` once the process is gone, ``_CANCELLED`` on cancel."""
        if not process.events.empty():
            return process.events.get_nowait()
        if cancel.cancelled:
            return _CANCELLED
        getter = asyncio.ensure_future(process.events.get())
        waiter = asyncio.ensure_future(cancel.wait())
        try:
            await asyncio.wait({getter, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
            if not getter.done():
                getter.cancel()
        return getter.result() if getter.done() and not getter.cancelled() else _CANCELLED

    async def _read(
        self, conversation: Conversation, cancel: CancelSignal, *, final: bool
    ) -> AsyncIterator[StreamEvent]:
        """Claude's next message: one that calls tools, or else the end of its turn."""
        process = conversation.process
        assert process is not None
        parts: dict[int, dict[str, Any]] = {}
        stop: str | None = None
        error: str | None = None
        started = False
        used = [0, 0]
        while True:
            event = await self._next(process, cancel)
            if event is _CANCELLED:
                raise _Cancelled
            if event is None:
                lines = list(process.errors)
                if conversation.session and not started and any(GONE_MARK in x for x in lines):
                    raise _Gone()
                detail = lines[-1] if lines else "no message"
                raise ProviderError(f"Claude Code stopped before it answered: {detail}")
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init":
                started = True
                if conversation.persist:
                    conversation.session = event.get("session_id") or conversation.session
            elif kind == "assistant" and event.get("error"):
                error = str(event["error"])
            elif kind == "stream_event":
                inner = event.get("event") or {}
                what = inner.get("type")
                if what == "message_start":
                    parts, stop = {}, None
                    usage = (inner.get("message") or {}).get("usage") or {}
                    used[0] += sum(
                        int(usage.get(name) or 0)
                        for name in (
                            "input_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        )
                    )
                elif what == "content_block_start":
                    block = inner.get("content_block") or {}
                    parts[inner.get("index", 0)] = {**block, "json": "", "text": ""}
                elif what == "content_block_delta":
                    part = parts.get(inner.get("index", 0))
                    delta = inner.get("delta") or {}
                    if part is None:
                        continue
                    if delta.get("type") == "text_delta" and part.get("type") == "text":
                        part["text"] += delta.get("text", "")
                        yield TextDelta(text=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        part["json"] += delta.get("partial_json", "")
                elif what == "message_delta":
                    stop = (inner.get("delta") or {}).get("stop_reason") or stop
                    used[1] += int((inner.get("usage") or {}).get("output_tokens") or 0)
                elif what == "message_stop":
                    calls = _calls(parts)
                    if stop == "tool_use" and calls:
                        conversation.waiting = [call.id for call in calls]
                        conversation.ended = False
                        yield UsageEvent(input_tokens=used[0], output_tokens=used[1])
                        for call in calls:
                            yield ToolCallEvent(call=call)
                        yield StopEvent(reason="tool_use")
                        if final:
                            raise _Cancelled
                        return
            elif kind == "result":
                conversation.ended = True
                if error or event.get("is_error"):
                    detail = str(event.get("result") or error or "unknown error")
                    raise ProviderError(f"Claude Code: {detail}")
                yield UsageEvent(input_tokens=used[0], output_tokens=used[1])
                yield StopEvent(reason="end")
                return


class _Cancelled(Exception):
    """The answer will not go on: the turn was cancelled, or it ignores the calls."""


_CANCELLED = object()


def _calls(parts: dict[int, dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for index in sorted(parts):
        part = parts[index]
        if part.get("type") != "tool_use":
            continue
        try:
            arguments = json.loads(part["json"] or "{}")
        except ValueError:
            arguments = {}
        name = str(part.get("name", ""))
        calls.append(
            ToolCall(
                id=str(part.get("id", "")),
                name=name.removeprefix(PREFIX),
                input=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls
