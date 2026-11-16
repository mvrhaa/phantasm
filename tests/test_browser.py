"""Zero's browser: the bridge to the Chrome extension, its endpoint and the browser tools."""

import asyncio
import json

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from zero.browser import BrowserBridge, BrowserError
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.protocol.messages import ImagePart
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.builtin.browser import BrowserClick, BrowserScreenshot, BrowserType, describe

EXTENSION = {"origin": "chrome-extension://abcdef"}


class FakeSocket:
    """An extension that answers every command the way ``reply`` says."""

    def __init__(self, reply) -> None:
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []
        self.reply = reply
        self.closed = False

    async def send_text(self, data: str) -> None:
        message = json.loads(data)
        self.sent.append(message)
        if "command" in message:
            await self.inbox.put(json.dumps({"id": message["id"], **self.reply(message)}))

    async def receive_text(self) -> str:
        text = await self.inbox.get()
        if text == "bye":
            raise WebSocketDisconnect()
        return text

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = True


async def test_a_call_gets_the_extensions_answer() -> None:
    bridge = BrowserBridge()
    socket = FakeSocket(lambda m: {"result": {"tab": 7, "url": m["args"]["url"]}})
    serving = asyncio.create_task(bridge.attach(socket))
    await asyncio.sleep(0)
    assert bridge.connected
    result = await bridge.call("open", {"url": "https://example.com"})
    assert result == {"tab": 7, "url": "https://example.com"}
    await socket.inbox.put(json.dumps({"type": "ping"}))
    await asyncio.sleep(0.01)
    assert socket.sent[-1] == {"type": "pong"}
    await socket.inbox.put("bye")
    await serving
    assert not bridge.connected


async def test_errors_and_absence_become_browser_errors(monkeypatch) -> None:
    monkeypatch.setattr("zero.browser.CONNECT_WAIT_S", 0.05)
    bridge = BrowserBridge()
    with pytest.raises(BrowserError, match="Chrome is not connected"):
        await bridge.call("tabs", {})
    socket = FakeSocket(lambda m: {"error": "tab 3 is not one of Zero's tabs"})
    serving = asyncio.create_task(bridge.attach(socket))
    await asyncio.sleep(0)
    with pytest.raises(BrowserError, match="not one of Zero's tabs"):
        await bridge.call("read", {"tab": 3})
    silent = FakeSocket(lambda m: {})
    silent.send_text = _swallow(silent)  # type: ignore[method-assign]
    second = asyncio.create_task(bridge.attach(silent))
    await asyncio.sleep(0)
    assert socket.closed
    cancel = CancelToken()
    asyncio.get_running_loop().call_later(0.05, cancel.cancel)
    with pytest.raises(BrowserError, match="stopped"):
        await bridge.call("read", {"tab": 1}, cancel)
    for task in (serving, second):
        task.cancel()


def _swallow(socket: FakeSocket):
    async def send_text(data: str) -> None:
        socket.sent.append(json.loads(data))

    return send_text


def test_only_an_extension_with_the_token_connects() -> None:
    with TestClient(create_app(Config(), None)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        for headers, query, code in (
            ({}, state.token, 4403),
            (EXTENSION, "wrong", 4401),
        ):
            with client.websocket_connect(f"/browser?token={query}", headers=headers) as ws:
                with pytest.raises(WebSocketDisconnect) as closed:
                    ws.receive_json()
                assert closed.value.code == code
        with client.websocket_connect(f"/browser?token={state.token}", headers=EXTENSION) as ws:
            future = client.portal.start_task_soon(state.browser.call, "tabs", {})
            asked = ws.receive_json()
            assert (asked["command"], asked["args"]) == ("tabs", {})
            ws.send_json({"id": asked["id"], "result": [{"tab": 1, "url": "u", "title": "t"}]})
            assert future.result(timeout=5) == [{"tab": 1, "url": "u", "title": "t"}]


def test_a_page_reads_as_text_and_numbered_elements() -> None:
    page = {
        "url": "https://shop.example",
        "title": "Shop",
        "text": "Welcome",
        "elements": [
            {"ref": 1, "tag": "a", "label": "Cart", "href": "https://shop.example/cart"},
            {"ref": 2, "tag": "input", "type": "search", "label": "Search", "value": "tea"},
        ],
    }
    text = describe(page)
    assert text.startswith("Shop — https://shop.example\n\nWelcome")
    assert "[1] a 'Cart' → https://shop.example/cart" in text
    assert "[2] input search 'Search' value='tea'" in text


def test_acting_asks_and_a_picture_is_an_image() -> None:
    bridge = BrowserBridge()
    assert BrowserClick(bridge).default_policy == "ask"
    assert BrowserType(bridge).default_policy == "ask"
    result = BrowserScreenshot(bridge).result({"mime": "image/jpeg", "data": "abc"})
    assert result.content[1] == ImagePart(mime="image/jpeg", data_b64="abc")
    assert BrowserScreenshot(bridge).result({}).is_error
