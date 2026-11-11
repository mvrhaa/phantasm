"""A fake "body": a client that declares a screenshot action and answers it with a picture.

    uv run python scripts/fake_body.py "what is on my screen?"
    uv run python scripts/fake_body.py --deny "delete my notes"

It reads the local token, connects, introduces itself with a character and a
``screenshot`` action, sends the message and prints the turn. ``action.call``
is answered with a small generated PNG, every ``approval.request`` with allow
(or deny with ``--deny``). It exits when the turn is finished.

It is also a minimal example of what a real client has to do.
"""

import argparse
import asyncio
import base64
import json
import struct
import uuid
import zlib
from typing import Any

import websockets

from zero.config import load_config
from zero.paths import token_path

SCREENSHOT = {
    "name": "screenshot",
    "description": "Capture the user's screen as a PNG image",
    "input_schema": {"type": "object", "properties": {"display": {"type": "integer"}}},
    "policy": "allow",
    "timeout_s": 15,
}
CHARACTER = {
    "name": "Zero",
    "persona": "A calm and curious companion.",
    "poses": ["idle", "wave", "think"],
    "moods": ["neutral", "happy", "sad"],
}


def tiny_png(width: int = 2, height: int = 2) -> bytes:
    """A small solid-blue PNG, built by hand so no image library is needed."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\x20\x60\xd0" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def show(event: dict[str, Any]) -> None:
    kind = event["type"]
    if kind == "text.delta":
        print(event["delta"], end="", flush=True)
        return
    details = {k: v for k, v in event.items() if k not in ("type", "session_id", "turn_id")}
    print(f"\n< {kind} {json.dumps(details, ensure_ascii=False)[:160]}")


async def answer(ws: websockets.ClientConnection, event: dict[str, Any], deny: bool) -> None:
    if event["type"] == "action.call" and event["action"] == "screenshot":
        picture = {
            "type": "image",
            "mime": "image/png",
            "data_b64": base64.b64encode(tiny_png()).decode(),
        }
        reply = {
            "type": "action.result",
            "call_id": event["call_id"],
            "ok": True,
            "content": [picture],
        }
        print("> action.result: a 2x2 blue PNG")
        await ws.send(json.dumps(reply))
    elif event["type"] == "action.call":
        reply = {
            "type": "action.result",
            "call_id": event["call_id"],
            "ok": False,
            "error": "unknown action",
        }
        await ws.send(json.dumps(reply))
    elif event["type"] == "approval.request":
        decision = "deny" if deny else "allow"
        print(f"> approval.respond: {decision}")
        await ws.send(
            json.dumps(
                {
                    "type": "approval.respond",
                    "approval_id": event["approval_id"],
                    "decision": decision,
                }
            )
        )


async def main(text: str, deny: bool) -> None:
    server = load_config().server
    host = f"[{server.host}]" if ":" in server.host else server.host
    token = token_path().read_text(encoding="utf-8").strip()
    async with websockets.connect(
        f"ws://{host}:{server.port}/ws", additional_headers={"Authorization": f"Bearer {token}"}
    ) as ws:
        client = {
            "app": "fake_body",
            "client_id": f"body-{uuid.uuid4().hex[:8]}",
            "display_name": "Fake body",
            "character": CHARACTER,
            "environment": "a test script in a terminal",
            "actions": [SCREENSHOT],
        }
        await ws.send(json.dumps({"type": "hello", "id": "hello", "protocol": 1, "client": client}))
        print(
            "< hello.ok" if json.loads(await ws.recv())["type"] == "hello.ok" else "! no hello.ok"
        )
        await ws.send(
            json.dumps({"type": "message.send", "id": "m1", "text": text, "source": "desktop"})
        )
        print(f"> message.send {text!r}")
        async for frame in ws:
            event = json.loads(frame)
            show(event)
            await answer(ws, event, deny)
            if event["type"] in ("turn.finished", "error"):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A fake client with a screenshot action.")
    parser.add_argument("text", help="the message to send")
    parser.add_argument("--deny", action="store_true", help="deny every approval request")
    arguments = parser.parse_args()
    try:
        asyncio.run(main(arguments.text, arguments.deny))
    except KeyboardInterrupt:
        pass
