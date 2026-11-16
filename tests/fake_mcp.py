"""A tiny MCP server over stdio for the tests: tools ``echo``, ``picture``, ``fail`` and ``die``.

``echo`` also asks the client for ``roots/list`` first, as some servers do.
"""

import json
import os
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the text back. Second sentence.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {"name": "picture", "description": "A tiny picture.", "inputSchema": {"type": "object"}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object"}},
    {"name": "die", "description": "Stops the server.", "inputSchema": {"type": "object"}},
]


def out(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def read() -> dict | None:
    line = sys.stdin.readline()
    return json.loads(line) if line else None


def main() -> None:
    sys.stderr.write("fake mcp up\n")
    while (message := read()) is not None:
        method, mid = message.get("method"), message.get("id")
        if mid is None:
            continue
        if method == "initialize":
            out(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1"},
                    },
                }
            )
        elif method == "tools/list":
            out({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = message["params"]["name"]
            arguments = message["params"].get("arguments") or {}
            if name == "echo":
                out({"jsonrpc": "2.0", "id": "srv-1", "method": "roots/list"})
                answer = read()
                roots = (answer or {}).get("result", {}).get("roots")
                text = f"{arguments.get('text')} {os.environ.get('FAKE_MCP_TAG', '')}".strip()
                out(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {"content": [{"type": "text", "text": f"{text} roots={roots}"}]},
                    }
                )
            elif name == "picture":
                out(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "content": [
                                {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
                                {
                                    "type": "resource",
                                    "resource": {"uri": "file:///x", "text": "inside"},
                                },
                            ]
                        },
                    }
                )
            elif name == "fail":
                out(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "content": [{"type": "text", "text": "it broke"}],
                            "isError": True,
                        },
                    }
                )
            elif name == "die":
                sys.exit(3)
        else:
            out({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "unknown"}})


if __name__ == "__main__":
    main()
