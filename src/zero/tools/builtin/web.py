"""Built-in ``web_fetch``: download a page and give the model its text.

Only http(s), at most 5 MB, 20 s and 5 redirects. HTML is reduced to its main
text with trafilatura; other text (plain, JSON, XML, …) comes back as it is.

A page can hide instructions meant for the model, so reaching this computer or
the local network is never silent: such an address needs the user's approval,
and a redirect to one is followed only within the host the user approved. A
host that changes its address between the check and the connection (DNS
rebinding) is not covered.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from pydantic import BaseModel, Field

from zero import __version__
from zero.core.cancellation import TurnCancelled
from zero.tools.base import Preview, Tool, ToolContext, ToolResult, error_result, text_result
from zero.tools.builtin.files import human_size

MAX_BYTES = 5 * 1024 * 1024
TIMEOUT_S = 20.0
MAX_REDIRECTS = 5
USER_AGENT = f"Zero/{__version__} (local assistant)"

Resolver = Callable[[str], Awaitable[list[str]]]
"""Gives every address a host name resolves to; raises ``OSError`` if there is none."""

_HTML_TYPES = ("text/html", "application/xhtml+xml")
_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/javascript")


class FetchRefused(Exception):
    """The page cannot be fetched; the message is written for the model."""


async def resolve_host(host: str) -> list[str]:
    """Every address ``host`` resolves to, through the system resolver."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def is_internal(address: str) -> bool:
    """True for an address on this computer or a local network rather than the internet."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


def url_host(url: str) -> str:
    """The host of an http(s) URL; raises ``FetchRefused`` for any other URL."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise FetchRefused(f"Only http and https URLs can be fetched, not {url!r}.")
    if parts.username is not None or parts.password is not None:
        raise FetchRefused("URLs with a user name or password in them are not fetched.")
    if not parts.hostname:
        raise FetchRefused(f"{url!r} has no host.")
    return parts.hostname


async def _read_body(response: httpx.Response) -> bytes:
    too_big = f"{response.url} is larger than {human_size(MAX_BYTES)}."
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BYTES:
        raise FetchRefused(too_big)
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > MAX_BYTES:
            raise FetchRefused(too_big)
    return bytes(body)


def _decoded(body: bytes, charset: str | None) -> str | bytes:
    """Text when the server named a usable charset; otherwise the bytes, for the parser to sniff."""
    if charset:
        try:
            return body.decode(charset, errors="replace")
        except LookupError:
            pass
    return body


def _extract_html(body: bytes, url: str, charset: str | None) -> tuple[str | None, str]:
    """The page title and its main text as Markdown; the whole text if no main part is found."""
    source = _decoded(body, charset)
    text = trafilatura.extract(source, url=url, output_format="markdown", include_comments=False)
    if not text:
        text = trafilatura.html2txt(source)
    metadata = trafilatura.extract_metadata(source)
    return (metadata.title if metadata is not None else None), text or ""


def _looks_like_html(body: bytes) -> bool:
    return body.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html"))


async def _page(asked: str, response: httpx.Response, body: bytes) -> ToolResult:
    final = str(response.url)
    if response.status_code >= 400:
        return error_result(
            f"{final} answered HTTP {response.status_code} {response.reason_phrase}."
        )
    kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
    title: str | None = None
    if kind in _HTML_TYPES or (not kind and _looks_like_html(body)):
        title, text = await asyncio.to_thread(_extract_html, body, final, response.charset_encoding)
    elif kind.startswith(_TEXT_TYPES) or kind.endswith(("+json", "+xml")):
        decoded = _decoded(body, response.charset_encoding)
        text = decoded if isinstance(decoded, str) else body.decode("utf-8", errors="replace")
    else:
        return error_result(
            f"{final} is {kind or 'of an unknown type'}, which web_fetch cannot read."
        )

    lines = [final if final == asked else f"{final} (redirected from {asked})"]
    if title:
        lines.append(f"Title: {title}")
    lines += ["", text.strip() or "(the page has no text)"]
    summary = f"{response.status_code}, {human_size(len(body))}"
    return text_result("\n".join(lines), summary=summary)


class WebFetchInput(BaseModel):
    url: str = Field(min_length=1, description="http or https URL of the page")


class WebFetch(Tool[WebFetchInput]):
    """Downloads a page and returns its text."""

    name = "web_fetch"
    description = (
        "Download a web page (http or https, up to 5 MB) and get its main text as Markdown. "
        "Plain text and JSON come back as they are. Addresses on this computer or the "
        "local network need the user's approval."
    )
    input_model = WebFetchInput
    default_policy = "allow"

    def __init__(
        self, transport: httpx.AsyncBaseTransport | None = None, resolver: Resolver = resolve_host
    ) -> None:
        self._transport = transport
        self._resolve = resolver

    def label(self, params: WebFetchInput) -> str:
        return params.url

    async def _internal(self, host: str) -> bool:
        try:
            ipaddress.ip_address(host)
            addresses = [host]
        except ValueError:
            try:
                addresses = await self._resolve(host)
            except OSError:
                raise FetchRefused(f"Cannot find the host {host}.") from None
        if not addresses:
            raise FetchRefused(f"Cannot find the host {host}.")
        return any(is_internal(address) for address in addresses)

    async def preview(self, params: WebFetchInput, ctx: ToolContext) -> Preview:
        try:
            host = url_host(params.url)
            internal = await self._internal(host)
        except FetchRefused as exc:
            return Preview(refusal=str(exc))
        if internal:
            return Preview(approval_reason=f"{host} points to this computer or the local network")
        return Preview()

    async def run(self, params: WebFetchInput, ctx: ToolContext) -> ToolResult:
        fetching = asyncio.create_task(self._fetch(params.url))
        cancelled = asyncio.create_task(ctx.cancel.wait())
        try:
            await asyncio.wait({fetching, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if not fetching.done():
                raise TurnCancelled()
            return fetching.result()
        finally:
            fetching.cancel()
            cancelled.cancel()

    async def _fetch(self, url: str) -> ToolResult:
        try:
            async with asyncio.timeout(TIMEOUT_S):
                return await self._follow(url)
        except FetchRefused as exc:
            return error_result(str(exc))
        except TimeoutError:
            return error_result(f"{url} did not answer within {TIMEOUT_S:g} s.")
        except httpx.HTTPError as exc:
            return error_result(f"Cannot fetch {url}: {str(exc) or type(exc).__name__}.")

    async def _follow(self, url: str) -> ToolResult:
        """GET the URL, following up to ``MAX_REDIRECTS`` redirects one by one."""
        approved_host = url_host(url)
        current = url
        async with httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=False,
            timeout=TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                async with client.stream("GET", current) as response:
                    if not response.is_redirect:
                        return await _page(url, response, await _read_body(response))
                    location = response.headers.get("location")
                if not location:
                    raise FetchRefused(f"{current} redirects without saying where to.")
                target = urljoin(current, location)
                host = url_host(target)
                if host != approved_host and await self._internal(host):
                    raise FetchRefused(
                        f"{current} redirects to {target}, which points to this computer or "
                        "the local network. Fetch that address directly if it is needed; "
                        "the user will be asked."
                    )
                current = target
        raise FetchRefused(f"{url} redirects more than {MAX_REDIRECTS} times.")
