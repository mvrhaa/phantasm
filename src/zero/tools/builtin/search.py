"""Built-in ``search_files``: a regex search with ripgrep, or in Python when rg is missing.

Both ways skip hidden names, binary files and symlinks, never report matches
from paths the sandbox protects, list files in path order and stop after
``MAX_MATCHES`` matches or ``SEARCH_TIMEOUT_S`` seconds. ripgrep also honours
``.gitignore`` files; the Python search does not read them.
"""

import asyncio
import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from zero import clock
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.tools.base import Preview, Tool, ToolContext, ToolResult, error_result, text_result
from zero.tools.builtin.files import PATH_HELP, SNIFF_BYTES, looks_secret
from zero.tools.permissions import PathDenied, PathGuard

MAX_MATCHES = 200
MAX_MATCH_CHARS = 300
SEARCH_TIMEOUT_S = 30.0
MAX_SCAN_BYTES = 10 * 1024 * 1024
RG_LINE_LIMIT = 64 * 1024 * 1024


class SearchFilesInput(BaseModel):
    pattern: str = Field(min_length=1, description="Regular expression to look for")
    path: str = Field(description=f"{PATH_HELP}; a folder to search in, or one file")
    glob: str | None = Field(
        default=None, description="Search only files whose path matches, e.g. *.py"
    )


@dataclass(frozen=True)
class Match:
    """One matching line."""

    path: str
    line: int
    text: str


def find_rg() -> str | None:
    """Where ripgrep is installed, or None."""
    return shutil.which("rg")


def _shown_path(path: Path, root: Path) -> str:
    return path.name if path == root else path.relative_to(root).as_posix()


def _allowed(guard: PathGuard, path: Path | str) -> bool:
    try:
        guard.check(path)
    except PathDenied:
        return False
    return True


def _render(matches: list[Match], stopped: str | None) -> ToolResult:
    lines = [f"{m.path}:{m.line}: {_cut(m.text)}" for m in matches] or ["No matches."]
    if stopped is not None:
        lines.append(f"[… {stopped}]")
    count = len(matches)
    return text_result("\n".join(lines), summary=f"{count} match{'' if count == 1 else 'es'}")


def _cut(text: str) -> str:
    text = text.rstrip("\r\n")
    return text if len(text) <= MAX_MATCH_CHARS else text[:MAX_MATCH_CHARS] + "…"


def _limit_note() -> str:
    return f"stopped after {MAX_MATCHES} matches; narrow the pattern, path or glob"


def _timeout_note() -> str:
    return f"search stopped after {SEARCH_TIMEOUT_S:g} s; narrow the path or glob"


def _rg_command(rg: str, params: SearchFilesInput, root: Path) -> list[str]:
    command = [rg, "--json", "--no-config", "--no-messages", "--sort", "path"]
    command += ["--max-filesize", str(MAX_SCAN_BYTES), "-e", params.pattern]
    if params.glob:
        command += ["--glob", params.glob]
    return [*command, "--", str(root)]


async def _collect(
    process: asyncio.subprocess.Process, root: Path, guard: PathGuard, matches: list[Match]
) -> bool:
    """Read rg's JSON lines into ``matches``; True once the match limit is reached."""
    assert process.stdout is not None
    allowed: dict[str, bool] = {}
    async for raw in process.stdout:
        event = json.loads(raw)
        if event.get("type") != "match":
            continue
        data = event["data"]
        path, text = data["path"].get("text"), data["lines"].get("text")
        if path is None or text is None:
            continue
        if path not in allowed:
            hidden = Path(path) != root and looks_secret(Path(path).name)
            allowed[path] = not hidden and await asyncio.to_thread(_allowed, guard, path)
        if not allowed[path]:
            continue
        matches.append(Match(_shown_path(Path(path), root), data["line_number"], text))
        if len(matches) >= MAX_MATCHES:
            return True
    return False


async def _search_rg(rg: str, params: SearchFilesInput, root: Path, ctx: ToolContext) -> ToolResult:
    process = await asyncio.create_subprocess_exec(
        *_rg_command(rg, params, root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=RG_LINE_LIMIT,
    )
    matches: list[Match] = []
    collecting = asyncio.create_task(_collect(process, root, ctx.paths, matches))
    cancelled = asyncio.create_task(ctx.cancel.wait())
    try:
        await asyncio.wait(
            {collecting, cancelled}, timeout=SEARCH_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
        )
        if not collecting.done():
            ctx.cancel.raise_if_cancelled()
            return _render(matches, _timeout_note())
        if collecting.result():
            return _render(matches, _limit_note())
        assert process.stderr is not None
        errors = (await process.stderr.read()).decode(errors="replace").strip()
        if await process.wait() == 2 and not matches:
            return error_result(f"Search failed: {errors[:500] or 'rg exited with code 2'}")
        return _render(matches, None)
    finally:
        cancelled.cancel()
        collecting.cancel()
        if process.returncode is None:
            process.kill()
            await process.wait()


def _files(root: Path, glob: str | None, guard: PathGuard) -> Iterator[Path]:
    """Files to search under ``root``, in path order; a file given directly is always searched."""
    if root.is_file():
        yield root
        return
    for folder, subfolders, names in os.walk(root):
        here = Path(folder)
        subfolders[:] = sorted(
            name
            for name in subfolders
            if not name.startswith(".")
            and not (here / name).is_symlink()
            and _allowed(guard, here / name)
        )
        for name in sorted(names):
            path = here / name
            if name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            if looks_secret(name):
                continue
            if glob and not PurePosixPath(path.relative_to(root).as_posix()).match(glob):
                continue
            if _allowed(guard, path):
                yield path


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    """Numbered lines of a text file; nothing for big, binary or unreadable files."""
    try:
        if path.stat().st_size > MAX_SCAN_BYTES:
            return
        with path.open("rb") as file:
            if b"\x00" in file.read(SNIFF_BYTES):
                return
        with path.open(encoding="utf-8", errors="replace", newline="") as file:
            yield from enumerate(file, start=1)
    except OSError:
        return


def _search_python(
    params: SearchFilesInput, root: Path, guard: PathGuard, cancel: CancelToken
) -> ToolResult:
    try:
        regex = re.compile(params.pattern)
    except re.error as exc:
        return error_result(f"Invalid pattern: {exc}.")
    deadline = clock.now() + timedelta(seconds=SEARCH_TIMEOUT_S)
    matches: list[Match] = []
    for path in _files(root, params.glob, guard):
        if cancel.cancelled:
            raise TurnCancelled()
        if clock.now() >= deadline:
            return _render(matches, _timeout_note())
        for number, text in _lines(path):
            if regex.search(text):
                matches.append(Match(_shown_path(path, root), number, text))
                if len(matches) >= MAX_MATCHES:
                    return _render(matches, _limit_note())
    return _render(matches, None)


def _root(params: SearchFilesInput, guard: PathGuard) -> Path | ToolResult:
    try:
        root = guard.check(params.path)
    except PathDenied as exc:
        return error_result(str(exc))
    if not root.exists():
        return error_result(f"{params.path} does not exist.")
    return root


class SearchFiles(Tool[SearchFilesInput]):
    """Finds lines that match a regular expression."""

    name = "search_files"
    description = (
        "Search files for a regular expression and get back path:line: text for each "
        "matching line, at most 200. Hidden names, binary files, symlinks and files that "
        "usually hold secrets are skipped. "
        "Use glob to narrow the files, e.g. *.py."
    )
    input_model = SearchFilesInput
    default_policy = "allow"

    def label(self, params: SearchFilesInput) -> str:
        return f"{params.pattern!r} in {params.path}"

    async def preview(self, params: SearchFilesInput, ctx: ToolContext) -> Preview:
        """Ask before searching one file that may hold secrets, as read_file does."""
        root = await asyncio.to_thread(_root, params, ctx.paths)
        if isinstance(root, ToolResult) or not await asyncio.to_thread(root.is_file):
            return Preview()
        if any(map(looks_secret, (Path(params.path).name, root.name))):
            return Preview(approval_reason=f"{params.path} may hold secrets")
        return Preview()

    async def run(self, params: SearchFilesInput, ctx: ToolContext) -> ToolResult:
        root = await asyncio.to_thread(_root, params, ctx.paths)
        if isinstance(root, ToolResult):
            return root
        rg = find_rg()
        if rg is not None:
            return await _search_rg(rg, params, root, ctx)
        return await asyncio.to_thread(_search_python, params, root, ctx.paths, ctx.cancel)
