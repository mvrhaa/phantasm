"""A stand-in for ``claude -p`` speaking stream-json, for the Claude Code provider's tests.

``FAKE_CLAUDE_DIR`` holds its state: ``turns.json`` is a queue of turns (each a
list of steps), ``log.jsonl`` gets its argv and every line it reads, ``sessions/``
the sessions it keeps. A step is ``{"text": ...}``, optionally with
``"tool": {"name", "input"}`` (it then waits for the tool's result), or
``{"error": ...}`` (the turn fails) or ``{"crash": ...}`` (the process dies).
"""

import json
import os
import sys
import uuid
from pathlib import Path

DIR = Path(os.environ["FAKE_CLAUDE_DIR"])
ARGV = sys.argv[1:]
PREFIX = "mcp__zero__"


def log(entry: dict) -> None:
    with (DIR / "log.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def out(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def arg(name: str) -> str | None:
    return ARGV[ARGV.index(name) + 1] if name in ARGV else None


def read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    message = json.loads(line)
    log({"in": message})
    return message


QUEUED: list[dict] = []
counter = 0


def ask(request: dict) -> dict:
    """Send a control request and wait for its answer (user messages meanwhile are queued)."""
    global counter
    counter += 1
    request_id = f"req-{counter}"
    out({"type": "control_request", "request_id": request_id, "request": request})
    while True:
        message = read()
        if message is None:
            sys.exit(0)
        if message.get("type") == "control_response":
            if message["response"].get("request_id") == request_id:
                return message["response"]
        elif message.get("type") == "user":
            QUEUED.append(message)


def next_turn() -> list[dict]:
    path = DIR / "turns.json"
    turns = json.loads(path.read_text())
    turn = turns.pop(0) if turns else [{"text": "(no more turns)"}]
    path.write_text(json.dumps(turns))
    return turn


def stream(event: dict) -> None:
    out({"type": "stream_event", "event": event})


def main() -> None:
    log({"argv": ARGV})
    tools = "--mcp-config" in ARGV
    session = arg("--resume")
    sessions = DIR / "sessions"
    sessions.mkdir(exist_ok=True)
    if session is not None and not (sessions / session).exists():
        sys.stderr.write(f"No conversation found with session ID: {session}\n")
        sys.exit(1)
    session = session or str(uuid.uuid4())
    if "--no-session-persistence" not in ARGV:
        (sessions / session).touch()
    first = read()
    assert first is not None and first["request"]["subtype"] == "initialize"
    if tools:
        ask(
            {
                "subtype": "mcp_message",
                "server_name": "zero",
                "message": {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            }
        )
        listed = ask(
            {
                "subtype": "mcp_message",
                "server_name": "zero",
                "message": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            }
        )
        log({"tools": listed["response"]["mcp_response"]["result"]["tools"]})
    out({"type": "control_response", "response": {"subtype": "success", "request_id": "init"}})
    announced = False
    uses = 0
    while True:
        message = QUEUED.pop(0) if QUEUED else read()
        if message is None:
            return
        if message.get("type") != "user":
            continue
        if not announced:
            out({"type": "system", "subtype": "init", "session_id": session})
            announced = True
        for step in next_turn():
            if "crash" in step:
                sys.stderr.write(step["crash"] + "\n")
                sys.exit(1)
            if "error" in step:
                out(
                    {
                        "type": "result",
                        "subtype": "error",
                        "is_error": True,
                        "result": step["error"],
                    }
                )
                break
            stream(
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 5}},
                }
            )
            stream(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            for word in step.get("text", "").split(" "):
                if word:
                    stream(
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": word + " "},
                        }
                    )
            tool = step.get("tool")
            if tool is None:
                stream(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 7},
                    }
                )
                stream({"type": "message_stop"})
                continue
            uses += 1
            use = f"toolu_{session[:4]}_{uses}"
            stream(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": use, "name": PREFIX + tool["name"]},
                }
            )
            stream(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(tool.get("input", {})),
                    },
                }
            )
            stream(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 3},
                }
            )
            stream({"type": "message_stop"})
            ask(
                {
                    "subtype": "can_use_tool",
                    "tool_name": PREFIX + tool["name"],
                    "input": tool.get("input", {}),
                }
            )
            result = ask(
                {
                    "subtype": "mcp_message",
                    "server_name": "zero",
                    "message": {
                        "jsonrpc": "2.0",
                        "id": 100 + uses,
                        "method": "tools/call",
                        "params": {
                            "name": tool["name"],
                            "arguments": tool.get("input", {}),
                            "_meta": {"claudecode/toolUseId": use},
                        },
                    },
                }
            )
            log({"result": result["response"]["mcp_response"]["result"]})
        else:
            out({"type": "result", "subtype": "success", "is_error": False, "result": "ok"})
        QUEUED.clear()


if __name__ == "__main__":
    main()
