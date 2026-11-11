"""Tests for the built-in web_fetch tool. There is no network: httpx gets a MockTransport."""

import asyncio
import socket
from collections.abc import Awaitable, Callable

import httpx
import pytest

from tests.turns import FakeApprover, registry, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.providers.base import ToolCall
from zero.tools.base import CANCELLED, ToolContext, ToolResult
from zero.tools.builtin import web
from zero.tools.builtin.web import USER_AGENT, WebFetch, is_internal
from zero.tools.permissions import PathGuard

ARTICLE = """<html><head><title>Trip to Rome</title></head><body>
<nav><a href="/">Home</a> <a href="/shop">Shop</a> Login</nav>
<div class="cookie">We use cookies. Accept?</div>
<article><h1>Trip to Rome</h1>
<p>Rome is the capital of Italy and has many old buildings worth seeing.</p>
<p>The best time to visit is spring, when the weather is warm but not hot.</p>
<script>alert(1)</script></article>
<footer>Copyright 2026 Example Inc. All rights reserved.</footer></body></html>"""

HOSTS = {
    "example.com": ["93.184.216.34"],
    "news.example": ["93.184.216.35"],
    "router.home": ["192.168.1.1"],
    "localhost": ["127.0.0.1", "::1"],
    "split.example": ["93.184.216.36", "10.0.0.5"],
}

Handler = Callable[[httpx.Request], Awaitable[httpx.Response] | httpx.Response]


async def fake_resolve(host: str) -> list[str]:
    if host not in HOSTS:
        raise socket.gaierror(f"unknown host {host}")
    return HOSTS[host]


def html(body: str = ARTICLE, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status, content=body.encode(), headers={"content-type": "text/html", **headers}
    )


class Web:
    """A fake internet: answers requests with ``handler`` and remembers them."""

    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx.Request] = []

        async def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            answer = handler(request)
            return await answer if asyncio.iscoroutine(answer) else answer  # type: ignore[return-value]

        self.tool = WebFetch(transport=httpx.MockTransport(record), resolver=fake_resolve)

    async def fetch(
        self,
        url: str,
        approver: FakeApprover | None = None,
        cancel: CancelToken | None = None,
        config: Config | None = None,
    ) -> ToolResult:
        config = config or Config()
        ctx = ToolContext(
            session_id="s1",
            turn_id="t1",
            client_id="c1",
            cancel=cancel or CancelToken(),
            config=config,
            paths=PathGuard.from_config(config.permissions),
            approvals=approver if approver is not None else FakeApprover("allow"),
        )
        call = ToolCall(id="f", name="web_fetch", input={"url": url})
        async with asyncio.timeout(10):
            return await registry(self.tool).execute(call, ctx)


# --- Addresses ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "internal"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("10.1.2.3", True),
        ("172.16.0.1", True),
        ("192.168.1.1", True),
        ("169.254.169.254", True),
        ("100.64.0.1", True),
        ("0.0.0.0", True),
        ("::ffff:127.0.0.1", True),
        ("fe80::1%en0", True),
        ("224.0.0.1", True),
        ("93.184.216.34", False),
        ("8.8.8.8", False),
        ("2606:2800:220:1:248:1893:25c8:1946", False),
    ],
)
def test_is_internal(address: str, internal: bool) -> None:
    assert is_internal(address) is internal


# --- Pages -------------------------------------------------------------------


async def test_html_page_comes_back_as_its_main_text() -> None:
    site = Web(lambda request: html())

    result = await site.fetch("https://example.com/rome")

    text = text_of(result)
    assert text.startswith("https://example.com/rome\nTitle: Trip to Rome\n\n")
    assert "Rome is the capital of Italy" in text
    assert "spring" in text
    for junk in ("Shop", "cookies", "Copyright", "alert"):
        assert junk not in text
    assert (result.summary, result.is_error) == (f"200, {web.human_size(len(ARTICLE))}", False)


async def test_request_names_zero_and_nothing_personal() -> None:
    site = Web(lambda request: html())

    await site.fetch("https://example.com/")

    agent = site.requests[0].headers["user-agent"]
    assert agent == USER_AGENT
    assert agent.startswith("Zero/") and "@" not in agent


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("text/plain", "just text\nsecond line"),
        ("application/json", '{"answer": 42}'),
        ("application/problem+json", '{"title": "nope"}'),
        ("text/markdown; charset=utf-8", "# Notes\n- one"),
    ],
)
async def test_other_text_comes_back_as_it_is(content_type: str, body: str) -> None:
    site = Web(
        lambda request: httpx.Response(200, text=body, headers={"content-type": content_type})
    )

    text = text_of(await site.fetch("https://example.com/data"))

    assert text == f"https://example.com/data\n\n{body}"


async def test_charset_from_the_header_is_used() -> None:
    body = (
        "<html><body><article>"
        "<p>Привет, это длинный абзац текста на русском языке, чтобы его выделить.</p>"
        "</article></body></html>"
    )
    site = Web(
        lambda request: httpx.Response(
            200,
            content=body.encode("windows-1251"),
            headers={"content-type": "text/html; charset=windows-1251"},
        )
    )

    assert "Привет, это длинный абзац" in text_of(await site.fetch("https://example.com/ru"))


async def test_charset_from_the_page_itself_is_used() -> None:
    body = (
        '<html><head><meta charset="windows-1251"></head><body><article>'
        "<p>Привет, это длинный абзац текста на русском языке, чтобы его выделить.</p>"
        "</article></body></html>"
    )
    site = Web(
        lambda request: httpx.Response(
            200, content=body.encode("windows-1251"), headers={"content-type": "text/html"}
        )
    )

    assert "Привет, это длинный абзац" in text_of(await site.fetch("https://example.com/ru"))


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "application/zip"])
async def test_unreadable_types_are_errors(content_type: str) -> None:
    site = Web(
        lambda request: httpx.Response(200, content=b"%PDF", headers={"content-type": content_type})
    )

    result = await site.fetch("https://example.com/file")

    assert result.is_error
    assert (
        text_of(result)
        == f"https://example.com/file is {content_type}, which web_fetch cannot read."
    )


async def test_http_error_status_is_an_error() -> None:
    site = Web(lambda request: html("gone", status=404))

    result = await site.fetch("https://example.com/missing")

    assert (text_of(result), result.is_error) == (
        "https://example.com/missing answered HTTP 404 Not Found.",
        True,
    )


# --- Limits ------------------------------------------------------------------


async def test_declared_size_over_the_limit_is_refused_before_reading() -> None:
    site = Web(lambda request: html(**{"content-length": str(web.MAX_BYTES + 1)}))

    assert text_of(await site.fetch("https://example.com/big")) == (
        "https://example.com/big is larger than 5.0 MB."
    )


async def test_body_over_the_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "MAX_BYTES", 100)

    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[override]
            while True:
                yield b"x" * 64

    site = Web(
        lambda request: httpx.Response(
            200, stream=Endless(), headers={"content-type": "text/plain"}
        )
    )

    result = await site.fetch("https://example.com/endless")

    assert text_of(result) == "https://example.com/endless is larger than 100 B."


async def test_slow_server_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "TIMEOUT_S", 0.2)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return html()

    result = await Web(slow).fetch("https://example.com/slow")

    assert text_of(result) == "https://example.com/slow did not answer within 0.2 s."


async def test_connection_failure_is_an_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = await Web(refuse).fetch("https://example.com/")

    assert text_of(result) == "Cannot fetch https://example.com/: connection refused."


async def test_cancel_stops_the_download() -> None:
    cancel = CancelToken()

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return html()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.2)
        cancel.cancel()

    result, _ = await asyncio.gather(
        Web(slow).fetch("https://example.com/", cancel=cancel), cancel_soon()
    )

    assert (text_of(result), result.cancelled) == (CANCELLED, True)


# --- Redirects ---------------------------------------------------------------


async def test_redirects_are_followed_and_shown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        if request.url.path == "/new":
            return httpx.Response(302, headers={"location": "https://news.example/rome"})
        return html()

    site = Web(handler)

    text = text_of(await site.fetch("https://example.com/old"))

    assert (
        text.splitlines()[0]
        == "https://news.example/rome (redirected from https://example.com/old)"
    )
    assert [str(r.url) for r in site.requests] == [
        "https://example.com/old",
        "https://example.com/new",
        "https://news.example/rome",
    ]


async def test_too_many_redirects_are_an_error() -> None:
    site = Web(lambda request: httpx.Response(302, headers={"location": "/again"}))

    result = await site.fetch("https://example.com/loop")

    assert text_of(result) == "https://example.com/loop redirects more than 5 times."
    assert len(site.requests) == web.MAX_REDIRECTS + 1


async def test_redirect_without_a_location_is_an_error() -> None:
    result = await Web(lambda request: httpx.Response(302)).fetch("https://example.com/")

    assert text_of(result) == "https://example.com/ redirects without saying where to."


@pytest.mark.parametrize(
    "target",
    ["http://192.168.1.1/admin", "http://router.home/", "http://localhost:2375/containers/json"],
)
async def test_redirect_from_the_internet_to_the_local_network_is_refused(target: str) -> None:
    site = Web(lambda request: httpx.Response(302, headers={"location": target}))

    result = await site.fetch("https://example.com/innocent")

    assert result.is_error
    assert text_of(result).startswith(
        f"https://example.com/innocent redirects to {target}, which points"
    )
    assert len(site.requests) == 1


async def test_redirect_within_an_approved_local_host_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "/login"})
        return httpx.Response(200, text="login page", headers={"content-type": "text/plain"})

    site = Web(handler)

    text = text_of(await site.fetch("http://localhost:3000/"))

    assert text.endswith("login page")


# --- What needs approval -----------------------------------------------------


async def test_public_page_needs_no_approval() -> None:
    approver = FakeApprover("allow")

    await Web(lambda request: html()).fetch("https://example.com/", approver)

    assert approver.asked == []


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("http://localhost:3000/", "localhost"),
        ("http://127.0.0.1:47400/health", "127.0.0.1"),
        ("http://router.home/", "router.home"),
        ("http://[::1]/", "::1"),
        ("http://169.254.169.254/latest/meta-data/", "169.254.169.254"),
        ("https://split.example/", "split.example"),
    ],
)
async def test_local_address_needs_approval_even_with_bypass(url: str, host: str) -> None:
    approver = FakeApprover("allow")
    config = Config()
    config.permissions.bypass_approvals = True

    await Web(lambda request: html()).fetch(url, approver, config=config)

    assert [ask.reason for ask in approver.asked] == [
        f"{host} points to this computer or the local network"
    ]


async def test_denied_local_address_is_not_fetched() -> None:
    site = Web(lambda request: html())

    result = await site.fetch("http://localhost:2375/containers/json", FakeApprover("deny"))

    assert result.denied
    assert site.requests == []


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (
            "file:///etc/passwd",
            "Only http and https URLs can be fetched, not 'file:///etc/passwd'.",
        ),
        ("ftp://example.com/x", "Only http and https URLs can be fetched"),
        ("example.com", "Only http and https URLs can be fetched"),
        (
            "https://user:secret@example.com/",
            "URLs with a user name or password in them are not fetched.",
        ),
        ("https://nowhere.invalid/", "Cannot find the host nowhere.invalid."),
        ("http:///path", "'http:///path' has no host."),
    ],
)
async def test_impossible_urls_are_refused_without_asking(url: str, message: str) -> None:
    approver = FakeApprover("allow")
    site = Web(lambda request: html())

    result = await site.fetch(url, approver)

    assert result.is_error
    assert text_of(result).startswith(message)
    assert approver.asked == []
    assert site.requests == []


def test_label_and_policy() -> None:
    tool = WebFetch()

    assert tool.label(web.WebFetchInput(url="https://example.com/a")) == "https://example.com/a"
    assert tool.default_policy == "allow"
