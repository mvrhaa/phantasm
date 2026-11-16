"""Browser tools: the user's own Chrome through Zero's extension (``zero/browser.py``).

Looking needs no approval (open, read, scroll, back, a picture, tabs); acting on
a page (click, type) asks the user unless they allowed it. Every tool works only
in the tabs of Zero's own Chrome window.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from zero.browser import BrowserBridge, BrowserError
from zero.protocol.messages import ImagePart, TextPart
from zero.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

READ_CHARS = 20_000


class NoInput(BaseModel):
    pass


class TabInput(BaseModel):
    tab: int = Field(description="The tab's id, as browser_open returned it")


class OpenInput(BaseModel):
    url: str = Field(min_length=1, description="The address to open")
    tab: int | None = Field(default=None, description="Open it in this tab instead of a new one")


class RefInput(TabInput):
    ref: int = Field(ge=1, description="The element's number from browser_read")


class TypeInput(RefInput):
    text: str = Field(description="What to put into the field")
    submit: bool = Field(default=False, description="Submit the form (Enter) after typing")


class ScrollInput(TabInput):
    direction: Literal["down", "up"] = "down"


def _page(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    return f"Tab {result.get('tab')}: {result.get('title') or '(no title)'} — {result.get('url')}"


def describe(page: dict[str, Any]) -> str:
    """A page as the model reads it: address, title, text, then numbered elements."""
    lines = [f"{page.get('title') or '(no title)'} — {page.get('url')}", ""]
    lines.append((page.get("text") or "")[:READ_CHARS])
    elements = page.get("elements") or []
    if elements:
        lines += ["", "Elements (use their numbers with browser_click and browser_type):"]
        for item in elements:
            kind = item.get("tag", "")
            if item.get("type"):
                kind += f" {item['type']}"
            line = f"[{item.get('ref')}] {kind} {item.get('label')!r}"
            if item.get("href"):
                line += f" → {item['href']}"
            if item.get("value"):
                line += f" value={item['value']!r}"
            lines.append(line)
    return "\n".join(lines)


class _BrowserTool[InputT: BaseModel](Tool[InputT]):
    default_policy = "allow"
    core = False
    command = ""

    def __init__(self, bridge: BrowserBridge) -> None:
        self._bridge = bridge

    async def _call(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        return await self._bridge.call(self.command, args, ctx.cancel)

    async def run(self, params: InputT, ctx: ToolContext) -> ToolResult:
        try:
            result = await self._call(params.model_dump(exclude_none=True), ctx)
        except BrowserError as exc:
            return error_result(str(exc))
        return self.result(result)

    def result(self, result: Any) -> ToolResult:
        return text_result(_page(result))


class BrowserOpen(_BrowserTool[OpenInput]):
    name = "browser_open"
    command = "open"
    description = (
        "Open a web page in the user's own Chrome (signed in as they are), in Zero's own "
        "window. Returns the tab's id; read the page with browser_read."
    )
    input_model = OpenInput

    def label(self, params: OpenInput) -> str:
        return params.url


class BrowserRead(_BrowserTool[TabInput]):
    name = "browser_read"
    command = "read"
    description = "Read a tab: its text and its links, buttons and fields, numbered."
    input_model = TabInput

    def result(self, result: Any) -> ToolResult:
        return text_result(describe(result) if isinstance(result, dict) else str(result))


class BrowserClick(_BrowserTool[RefInput]):
    name = "browser_click"
    command = "click"
    description = "Click an element of a tab by its number from browser_read. Read again after."
    input_model = RefInput
    default_policy = "ask"

    def label(self, params: RefInput) -> str:
        return f"click [{params.ref}] in tab {params.tab}"


class BrowserType(_BrowserTool[TypeInput]):
    name = "browser_type"
    command = "type"
    description = (
        "Fill a field of a tab by its number from browser_read; submit=true presses Enter."
    )
    input_model = TypeInput
    default_policy = "ask"

    def label(self, params: TypeInput) -> str:
        return f"type into [{params.ref}] in tab {params.tab}"


class BrowserScroll(_BrowserTool[ScrollInput]):
    name = "browser_scroll"
    command = "scroll"
    description = "Scroll a tab by a screen, down or up."
    input_model = ScrollInput


class BrowserBack(_BrowserTool[TabInput]):
    name = "browser_back"
    command = "back"
    description = "Go back to the previous page of a tab."
    input_model = TabInput


class BrowserScreenshot(_BrowserTool[TabInput]):
    name = "browser_screenshot"
    command = "screenshot"
    description = "A picture of what a tab shows."
    input_model = TabInput

    def result(self, result: Any) -> ToolResult:
        if not isinstance(result, dict) or not result.get("data"):
            return error_result("Chrome gave no picture of the tab.")
        image = ImagePart(mime=result.get("mime") or "image/jpeg", data_b64=result["data"])
        return ToolResult(content=[TextPart(text="The tab looks like this:"), image])


class BrowserTabs(_BrowserTool[NoInput]):
    name = "browser_tabs"
    command = "tabs"
    description = "The tabs of Zero's Chrome window: id, title, address."
    input_model = NoInput

    def result(self, result: Any) -> ToolResult:
        tabs = result if isinstance(result, list) else []
        return text_result("\n".join(_page(tab) for tab in tabs) or "No tabs are open.")


class BrowserClose(_BrowserTool[TabInput]):
    name = "browser_close"
    command = "close"
    description = "Close a tab of Zero's Chrome window."
    input_model = TabInput

    def result(self, result: Any) -> ToolResult:
        return text_result("Closed.")


def browser_tools(bridge: BrowserBridge) -> list[Tool[Any]]:
    return [
        tool(bridge)
        for tool in (
            BrowserOpen,
            BrowserRead,
            BrowserClick,
            BrowserType,
            BrowserScroll,
            BrowserBack,
            BrowserScreenshot,
            BrowserTabs,
            BrowserClose,
        )
    ]
