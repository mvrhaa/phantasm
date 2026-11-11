"""Built-in file tools: ``read_file`` and ``list_dir``.

All file system work runs in a worker thread, and every path goes through the
sandbox (``ctx.paths``) before it is touched.
"""

import asyncio
import base64
import os
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field

from zero.protocol.messages import ImagePart
from zero.tools.base import (
    MAX_OUTPUT_CHARS,
    Tool,
    ToolContext,
    ToolResult,
    error_result,
    text_result,
)
from zero.tools.permissions import PathDenied

DEFAULT_READ_LINES = 2000
MAX_LINE_CHARS = 2000
TEXT_BUDGET = MAX_OUTPUT_CHARS - 1000  # room for the note about the next page
MAX_IMAGE_BYTES = 5 * 1024 * 1024
SNIFF_BYTES = 8192
MAX_LIST_DEPTH = 5
MAX_LIST_ENTRIES = 500

PATH_HELP = "Absolute path, or a path that starts with ~"

_IMAGE_SIGNATURES = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def human_size(size: int) -> str:
    """A file size the way people read it: ``512 B``, ``1.5 KB``, ``2.0 MB``."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    return f"{size} B" if unit == 0 else f"{value:.1f} {units[unit]}"


def looks_secret(name: str) -> bool:
    """True for file names that usually hold secrets: ``.env*``, ``id_*`` and ``*.pem``."""
    name = unicodedata.normalize("NFC", name).casefold()
    return name.startswith((".env", "id_")) or name.endswith(".pem")


def _image_mime(head: bytes) -> str | None:
    for signature, mime in _IMAGE_SIGNATURES:
        if head.startswith(signature):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


# --- read_file ---------------------------------------------------------------


class ReadFileInput(BaseModel):
    path: str = Field(description=PATH_HELP)
    offset: int = Field(default=1, ge=1, description="Line to start from, counting from 1")
    limit: int = Field(default=DEFAULT_READ_LINES, ge=1, description="Most lines to return")


def _read_text(path: Path, shown: str, offset: int, limit: int) -> ToolResult:
    lines: list[str] = []
    used = 0
    total = 0
    next_line: int | None = None
    with path.open(encoding="utf-8", errors="replace", newline="") as file:
        for total, line in enumerate(file, start=1):
            if total < offset:
                continue
            text = line.rstrip("\r\n")
            if len(text) > MAX_LINE_CHARS:
                text = text[:MAX_LINE_CHARS] + " [… line cut]"
            numbered = f"{total:>6}\t{text}"
            if len(lines) == limit or (lines and used + len(numbered) > TEXT_BUDGET):
                next_line = total
                break
            lines.append(numbered)
            used += len(numbered) + 1

    if not lines:
        if total == 0:
            return text_result("(empty file)", summary="empty")
        return error_result(f"{shown} has only {total} lines; offset {offset} is past the end.")
    body = "\n".join(lines)
    if next_line is not None:
        body += f"\n[… more lines follow; read on with offset={next_line}]"
    return text_result(body, summary=f"{len(lines)} lines")


def _read_image(path: Path, shown: str, mime: str) -> ToolResult:
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        limit = human_size(MAX_IMAGE_BYTES)
        return error_result(f"{shown} is an image of {human_size(size)}; the limit is {limit}.")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return ToolResult(
        content=[ImagePart(mime=mime, data_b64=data)], summary=f"image, {human_size(size)}"
    )


class ReadFile(Tool[ReadFileInput]):
    """Reads text with line numbers, or an image as an image."""

    name = "read_file"
    description = (
        "Read a file. Text comes back with line numbers; use offset and limit to page "
        "through long files. PNG, JPEG, GIF and WebP images come back as images. "
        "Other binary files cannot be read."
    )
    input_model = ReadFileInput
    default_policy = "allow"

    def label(self, params: ReadFileInput) -> str:
        return params.path

    async def approval_reason(self, params: ReadFileInput, ctx: ToolContext) -> str | None:
        return await asyncio.to_thread(self._secret_reason, params, ctx)

    def _secret_reason(self, params: ReadFileInput, ctx: ToolContext) -> str | None:
        try:
            resolved = ctx.paths.check(params.path)
        except PathDenied:
            return None  # run() refuses it with the sandbox's own reason
        names = [Path(params.path).name, resolved.name]
        return f"{params.path} may hold secrets" if any(map(looks_secret, names)) else None

    async def run(self, params: ReadFileInput, ctx: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._read, params, ctx)

    def _read(self, params: ReadFileInput, ctx: ToolContext) -> ToolResult:
        shown = params.path
        try:
            path = ctx.paths.check(params.path)
            if not path.exists():
                return error_result(f"{shown} does not exist.")
            if path.is_dir():
                return error_result(f"{shown} is a folder; use list_dir to see what is in it.")
            if not path.is_file():
                return error_result(f"{shown} is not a regular file.")
            with path.open("rb") as file:
                head = file.read(SNIFF_BYTES)
            mime = _image_mime(head)
            if mime is not None:
                return _read_image(path, shown, mime)
            if b"\x00" in head:
                return error_result(f"{shown} is a binary file and cannot be read as text.")
            return _read_text(path, shown, params.offset, params.limit)
        except PathDenied as exc:
            return error_result(str(exc))
        except OSError as exc:
            return error_result(f"Cannot read {shown}: {exc.strerror or exc}.")


# --- list_dir ----------------------------------------------------------------


class ListDirInput(BaseModel):
    path: str = Field(description=PATH_HELP)
    depth: int = Field(
        default=1, ge=1, le=MAX_LIST_DEPTH, description="Levels to show; 1 is this folder only"
    )
    show_hidden: bool = Field(default=False, description="Include names that start with a dot")


class _Listing:
    """Builds the tree text of ``list_dir``, stopping after ``MAX_LIST_ENTRIES`` entries."""

    def __init__(self, params: ListDirInput, ctx: ToolContext) -> None:
        self.params = params
        self.ctx = ctx
        self.lines: list[str] = []
        self.entries = 0
        self.full = False

    def walk(self, folder: Path, level: int) -> None:
        indent = "  " * level
        try:
            with os.scandir(folder) as scan:
                found = [e for e in scan if self.params.show_hidden or not e.name.startswith(".")]
        except OSError as exc:
            self.lines.append(f"{indent}[cannot list: {exc.strerror or exc}]")
            return
        found.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.casefold()))
        if not found and level == 1:
            self.lines.append(f"{indent}(empty)")
        for entry in found:
            if self.entries == MAX_LIST_ENTRIES:
                self.full = True
                return
            self.entries += 1
            self._add(entry, indent, level)
            if self.full:
                return

    def _add(self, entry: os.DirEntry[str], indent: str, level: int) -> None:
        if entry.is_symlink():
            self.lines.append(f"{indent}{entry.name} -> {os.readlink(entry.path)}")
        elif entry.is_dir(follow_symlinks=False):
            if level < self.params.depth:
                try:
                    self.ctx.paths.check(entry.path)
                except PathDenied:
                    self.lines.append(f"{indent}{entry.name}/  (protected)")
                    return
                self.lines.append(f"{indent}{entry.name}/")
                self.walk(Path(entry.path), level + 1)
            else:
                self.lines.append(f"{indent}{entry.name}/")
        else:
            size = human_size(entry.stat(follow_symlinks=False).st_size)
            self.lines.append(f"{indent}{entry.name}  {size}")


class ListDir(Tool[ListDirInput]):
    """Shows a folder as a tree with file sizes."""

    name = "list_dir"
    description = (
        "List a folder as a tree: subfolders end with /, files show their size, "
        "symlinks show where they point. depth sets how many levels to show. "
        "Names that start with a dot are left out unless show_hidden is true."
    )
    input_model = ListDirInput
    default_policy = "allow"

    def label(self, params: ListDirInput) -> str:
        return params.path

    async def run(self, params: ListDirInput, ctx: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._list, params, ctx)

    def _list(self, params: ListDirInput, ctx: ToolContext) -> ToolResult:
        shown = params.path
        try:
            root = ctx.paths.check(params.path)
        except PathDenied as exc:
            return error_result(str(exc))
        if not root.exists():
            return error_result(f"{shown} does not exist.")
        if not root.is_dir():
            return error_result(f"{shown} is a file; use read_file to read it.")

        listing = _Listing(params, ctx)
        listing.walk(root, 1)
        lines = [f"{shown.rstrip('/')}/", *listing.lines]
        if listing.full:
            lines.append(f"[… stopped after {MAX_LIST_ENTRIES} entries; list a subfolder]")
        return text_result("\n".join(lines), summary=f"{listing.entries} entries")
