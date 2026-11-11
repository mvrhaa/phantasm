"""A tiny interactive client for trying the protocol by hand.

    uv run python scripts/ws_client.py

It reads the local token, connects, sends ``hello`` and then sends every line
you type as a message. A line may be full JSON, or a shortcut:

    list                     session.list
    create [title]           session.create
    rename <id> <title>      session.rename
    delete <id>              session.delete
    history <id>             history.get
    ping                     ping

Everything the server sends is printed. Ctrl+D or Ctrl+C quits.
"""

import asyncio
import json
import sys
import uuid

import websockets

from zero.config import load_config
from zero.paths import token_path


def build(line: str) -> dict[str, object] | None:
    """Turn a typed line into a protocol message, or None if it is not understood."""
    if line.startswith("{"):
        return dict(json.loads(line))
    command, _, rest = line.partition(" ")
    rest = rest.strip()
    if command == "list":
        return {"type": "session.list"}
    if command == "create":
        return {"type": "session.create", "title": rest or None}
    if command == "rename":
        session_id, _, title = rest.partition(" ")
        return {"type": "session.rename", "session_id": session_id, "title": title}
    if command == "delete":
        return {"type": "session.delete", "session_id": rest}
    if command == "history":
        return {"type": "history.get", "session_id": rest}
    if command == "ping":
        return {"type": "ping"}
    return None


async def printer(ws: websockets.ClientConnection) -> None:
    async for frame in ws:
        print("<", frame)


async def main() -> None:
    server = load_config().server
    host = f"[{server.host}]" if ":" in server.host else server.host
    token = token_path().read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(
        f"ws://{host}:{server.port}/ws", additional_headers=headers
    ) as ws:
        reader = asyncio.create_task(printer(ws))
        hello = {
            "type": "hello",
            "id": "hello",
            "protocol": 1,
            "client": {"app": "ws_client", "client_id": f"cli-{uuid.uuid4().hex[:8]}"},
        }
        await ws.send(json.dumps(hello))
        loop = asyncio.get_running_loop()
        while line := await loop.run_in_executor(None, sys.stdin.readline):
            line = line.strip()
            if not line:
                continue
            try:
                message = build(line)
            except ValueError as exc:
                print("! bad JSON:", exc)
                continue
            if message is None:
                print("! unknown command, see the top of scripts/ws_client.py")
                continue
            message.setdefault("id", uuid.uuid4().hex[:8])
            print(">", json.dumps(message, ensure_ascii=False))
            await ws.send(json.dumps(message))
        await asyncio.sleep(0.3)  # let the last answers arrive before closing
        reader.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
