"""Built-in tools that change files: ``write_file`` and ``edit_file``.

A change is shown to the user as a unified diff before it is made, and files
are replaced atomically, so a failed write never leaves half a file behind.
A change that would break the syntax of a Python, JSON or TOML file is refused
before the user is asked, so the model fixes it first.
"""

import ast
import asyncio
import contextlib
import difflib
import json
import os
import shutil
import tempfile
import tomllib
import warnings
from collections.abc import Callable
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
    old: str | None
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
    try:
        return Target(path, data.decode("utf-8"))
    except UnicodeDecodeError:
        return Target(path, None, binary=True)


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
        lines.insert(
            0, f"{shown} is a binary file or not UTF-8 text; it will be replaced with UTF-8 text."
        )
    return "\n".join(lines)


def diff_summary(diff: str) -> str:
    """``+N -M``: how many lines a diff adds and removes."""
    lines = diff.splitlines()
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    return f"+{added} -{removed}"


def _python_problem(text: str) -> str | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"
        except (ValueError, RecursionError) as exc:
            return str(exc) or type(exc).__name__
    return None


def _json_problem(text: str) -> str | None:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except RecursionError:
        return "nested too deeply"
    return None


def _toml_problem(text: str) -> str | None:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return str(exc)
    return None


_SYNTAX_CHECKS: dict[str, tuple[str, Callable[[str], str | None]]] = {
    ".py": ("Python", _python_problem),
    ".pyi": ("Python", _python_problem),
    ".json": ("JSON", _json_problem),
    ".toml": ("TOML", _toml_problem),
}


def syntax_problem(shown: str, target: Target, new: str) -> str | None:
    """Why ``new`` would break a Python, JSON or TOML file, for the model.

    None when the new text parses, the file is of another kind, or it did not
    parse before either (a half-fixed file may still be changed).
    """
    check = _SYNTAX_CHECKS.get(target.path.suffix.lower())
    if check is None:
        return None
    kind, problem_of = check
    problem = problem_of(new)
    if problem is None or (target.old is not None and problem_of(target.old) is not None):
        return None
    return (
        f"The change was not made: it would break the {kind} syntax of {shown} ({problem}). "
        "Fix the change and try again."
    )


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
        if (problem := syntax_problem(params.path, target, params.content)) is not None:
            return Preview(refusal=problem)
        diff = make_diff(params.path, target, params.content)
        return Preview(diff=truncate_text(diff, MAX_OUTPUT_CHARS))

    async def run(self, params: WriteFileInput, ctx: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._write, params, ctx)

    def _write(self, params: WriteFileInput, ctx: ToolContext) -> ToolResult:
        shown = params.path
        target = load_target(shown, ctx.paths)
        if isinstance(target, str):
            return error_result(target)
        if target.old == params.content:
            return text_result(
                f"{shown} already has this content; nothing was written.", summary="no changes"
            )
        if (problem := syntax_problem(shown, target, params.content)) is not None:
            return error_result(problem)
        summary = diff_summary(make_diff(shown, target, params.content))
        try:
            write_atomically(target.path, params.content)
        except OSError as exc:
            return error_result(f"Cannot write {shown}: {exc.strerror or exc}.")
        verb = "Wrote" if target.exists else "Created"
        return text_result(f"{verb} {shown} ({summary}).", summary=summary)


class EditFileInput(BaseModel):
    path: str = Field(description=PATH_HELP)
    old: str = Field(
        min_length=1,
        description="Exact text to replace; it must appear in the file exactly once",
    )
    new: str = Field(description="Text to put in its place")


def _occurrences(text: str, part: str) -> int:
    """How many times ``part`` appears in ``text``, overlapping matches included."""
    count = 0
    start = text.find(part)
    while start != -1:
        count += 1
        start = text.find(part, start + 1)
    return count


def plan_edit(params: EditFileInput, guard: PathGuard) -> tuple[Target, str] | str:
    """The file as it is and as it will be after the edit, or an error for the model.

    If the file uses Windows line endings (CRLF) and ``old`` does not match as
    it is, ``old`` and ``new`` are tried with CRLF line endings: read_file shows
    lines without the carriage return, so the model usually sends plain LF.
    """
    shown = params.path
    target = load_target(shown, guard)
    if isinstance(target, str):
        return target
    if not target.exists:
        return f"{shown} does not exist; use write_file to create it."
    if target.old is None:
        return f"{shown} is a binary file or not UTF-8 text and cannot be edited."
    if params.old == params.new:
        return "old and new are the same, so there is nothing to change."

    old, new = params.old, params.new
    count = _occurrences(target.old, old)
    if count == 0 and "\r\n" in target.old and "\n" in old:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        count = _occurrences(target.old, old)
    if count == 0:
        return (
            f"old was not found in {shown}. Copy it exactly as it is in the file, "
            "without the line numbers read_file shows."
        )
    if count > 1:
        return (
            f"old appears {count} times in {shown}; include more of the surrounding "
            "lines so that it matches exactly once."
        )
    edited = target.old.replace(old, new, 1)
    if (problem := syntax_problem(shown, target, edited)) is not None:
        return problem
    return target, edited


class EditFile(Tool[EditFileInput]):
    """Replaces one exact piece of text in a file."""

    name = "edit_file"
    description = (
        "Change part of a file: replace the exact text old with new. old must appear in "
        "the file exactly once; include enough surrounding lines to make it unique. "
        "The user usually sees the change as a diff and approves it first."
    )
    input_model = EditFileInput
    default_policy = "ask"

    def label(self, params: EditFileInput) -> str:
        return params.path

    async def preview(self, params: EditFileInput, ctx: ToolContext) -> Preview:
        plan = await asyncio.to_thread(plan_edit, params, ctx.paths)
        if isinstance(plan, str):
            return Preview(refusal=plan)
        target, edited = plan
        return Preview(diff=truncate_text(make_diff(params.path, target, edited), MAX_OUTPUT_CHARS))

    async def run(self, params: EditFileInput, ctx: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._edit, params, ctx)

    def _edit(self, params: EditFileInput, ctx: ToolContext) -> ToolResult:
        plan = plan_edit(params, ctx.paths)
        if isinstance(plan, str):
            return error_result(plan)
        target, edited = plan
        summary = diff_summary(make_diff(params.path, target, edited))
        try:
            write_atomically(target.path, edited)
        except OSError as exc:
            return error_result(f"Cannot write {params.path}: {exc.strerror or exc}.")
        return text_result(f"Edited {params.path} ({summary}).", summary=summary)
