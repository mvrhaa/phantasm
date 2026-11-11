"""Built-in tools that change files: ``write_file``.

A change is shown to the user as a unified diff before it is made, and files
are replaced atomically, so a failed write never leaves half a file behind.
"""

import asyncio
import contextlib
import difflib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from zero.tools.base import (
    MAX_OUTPUT_CHARS,
    Preview,
    Tool,
    ToolContext,
    ToolResult,
    error_result,
    text_result,
    truncate_text,
)
from zero.tools.builtin.files import PATH_HELP, SNIFF_BYTES
from zero.tools.permissions import PathDenied, PathGuard


@dataclass(frozen=True)
class Target:
    """A file about to be changed, as it is now."""

    path: Path
    old: str | None  # None when the file does not exist yet or holds binary data
    binary: bool = False

    @property
    def exists(self) -> bool:
        return self.old is not None or self.binary


def load_target(raw: str, guard: PathGuard) -> Target | str:
    """Check the path and read what the file holds now; a string is an error for the model."""
    try:
        path = guard.check(raw)
        if path.is_dir():
            return f"{raw} is a folder, not a file."
        if not path.exists():
            return Target(path, None)
        if not path.is_file():
            return f"{raw} is not a regular file."
        data = path.read_bytes()
    except PathDenied as exc:
        return str(exc)
    except OSError as exc:
        return f"Cannot read {raw}: {exc.strerror or exc}."
    if b"\x00" in data[:SNIFF_BYTES]:
        return Target(path, None, binary=True)
    return Target(path, data.decode("utf-8", errors="replace"))


def make_diff(shown: str, target: Target, new: str) -> str:
    """A unified diff from the file as it is to ``new``; a new file is all additions."""
    lines = list(
        difflib.unified_diff(
            target.old.splitlines() if target.old is not None else [],
            new.splitlines(),
            fromfile=shown if target.exists else "/dev/null",
            tofile=shown,
            lineterm="",
        )
    )
    if target.binary:
        lines.insert(0, f"Binary file {shown} will be replaced with text.")
    return "\n".join(lines)


def diff_summary(diff: str) -> str:
    """``+N -M``: how many lines a diff adds and removes."""
    lines = diff.splitlines()
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    return f"+{added} -{removed}"


def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8, creating missing folders.

    An existing file is replaced in one step (a temporary file is renamed over
    it) and keeps its permissions. A new file is created with the usual ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(text)
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


# --- write_file --------------------------------------------------------------


class WriteFileInput(BaseModel):
    path: str = Field(description=PATH_HELP)
    content: str = Field(description="The whole new content of the file")


class WriteFile(Tool[WriteFileInput]):
    """Creates a file or replaces everything in it."""

    name = "write_file"
    description = (
        "Create a file, or replace everything in an existing one, with content. Missing "
        "folders are created. The user usually sees the change as a diff and approves it "
        "first. To change a few lines of a big file, prefer edit_file."
    )
    input_model = WriteFileInput
    default_policy = "ask"

    def label(self, params: WriteFileInput) -> str:
        return params.path

    async def preview(self, params: WriteFileInput, ctx: ToolContext) -> Preview:
        target = await asyncio.to_thread(load_target, params.path, ctx.paths)
        if isinstance(target, str):
            return Preview(refusal=target)
        diff = make_diff(params.path, target, params.content)
        return Preview(diff=truncate_text(diff, MAX_OUTPUT_CHARS))

    async def run(self, params: WriteFileInput, ctx: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._write, params, ctx)

    def _write(self, params: WriteFileInput, ctx: ToolContext) -> ToolResult:
        shown = params.path
        target = load_target(shown, ctx.paths)  # again: the file may have changed meanwhile
        if isinstance(target, str):
            return error_result(target)
        if target.old == params.content:
            return text_result(
                f"{shown} already has this content; nothing was written.", summary="no changes"
            )
        summary = diff_summary(make_diff(shown, target, params.content))
        try:
            write_atomically(target.path, params.content)
        except OSError as exc:
            return error_result(f"Cannot write {shown}: {exc.strerror or exc}.")
        verb = "Wrote" if target.exists else "Created"
        return text_result(f"{verb} {shown} ({summary}).", summary=summary)
