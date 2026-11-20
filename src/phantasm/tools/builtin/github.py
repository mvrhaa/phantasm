"""GitHub through the user's own ``gh`` CLI: its login, no token in Phantasm's hands.

Reading asks nothing (``github_get``: any GET of the REST API; ``github_failed_log``:
the log of what failed in an Actions run). Changes ask the user first
(``github_api``: issues, comments, pull requests and any other call).
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

TIMEOUT_S = 60.0
INSTALL_FOLDERS = ("/usr/local/bin", "/opt/homebrew/bin", "~/.local/bin")
ENV = {"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "NO_COLOR": "1"}
NOT_INSTALLED = "The gh command is not installed; install GitHub CLI and run gh auth login."


def find_gh() -> str | None:
    found = shutil.which("gh")
    if found:
        return found
    for folder in INSTALL_FOLDERS:
        path = Path(folder).expanduser() / "gh"
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


async def run_gh(command: str | None, args: list[str], stdin: str | None = None) -> ToolResult:
    """Run gh and give its output, or its error as an error result."""
    if command is None:
        return error_result(NOT_INSTALLED)
    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        env={**os.environ, **ENV},
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            process.communicate(stdin.encode() if stdin is not None else None), TIMEOUT_S
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return error_result(f"gh did not finish in {TIMEOUT_S:.0f} s")
    text = out.decode("utf-8", "replace").strip()
    if process.returncode != 0:
        detail = err.decode("utf-8", "replace").strip() or text or f"exit {process.returncode}"
        return error_result(f"gh failed: {detail[-2000:]}")
    return text_result(text or "(no output)")


class GithubGetInput(BaseModel):
    path: str = Field(
        min_length=1,
        description="REST path, e.g. repos/OWNER/REPO/issues?state=open or search/code?q=...",
    )
    paginate: bool = Field(default=False, description="Fetch every page of a list")


class GithubGet(Tool[GithubGetInput]):
    name = "github_get"
    description = (
        "Read from GitHub's REST API as the user (their gh login): repositories, files, "
        "issues, pull requests with their comments and diffs, Actions runs, search. Use "
        "user/repos for the user's own repositories."
    )
    input_model = GithubGetInput
    default_policy = "allow"
    core = False

    def __init__(self, command: str | None = None) -> None:
        self._command = command

    def label(self, params: GithubGetInput) -> str:
        return f"GET {params.path}"

    async def run(self, params: GithubGetInput, ctx: ToolContext) -> ToolResult:
        args = ["api", params.path.lstrip("/"), "--method", "GET"]
        if params.paginate:
            args.append("--paginate")
        return await run_gh(self._command or find_gh(), args)


class GithubApiInput(BaseModel):
    method: Literal["POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(min_length=1, description="REST path, e.g. repos/OWNER/REPO/issues")
    body: dict[str, Any] | None = Field(default=None, description="JSON body of the request")


class GithubApi(Tool[GithubApiInput]):
    name = "github_api"
    description = (
        "Change something on GitHub as the user: open or comment on issues and pull "
        "requests, review, merge, any other REST call that writes. The user is asked first."
    )
    input_model = GithubApiInput
    default_policy = "ask"
    core = False

    def __init__(self, command: str | None = None) -> None:
        self._command = command

    def label(self, params: GithubApiInput) -> str:
        return f"{params.method} {params.path}"

    async def run(self, params: GithubApiInput, ctx: ToolContext) -> ToolResult:
        args = ["api", params.path.lstrip("/"), "--method", params.method]
        body = None
        if params.body is not None:
            args += ["--input", "-"]
            body = json.dumps(params.body, ensure_ascii=False)
        return await run_gh(self._command or find_gh(), args, stdin=body)


class GithubFailedLogInput(BaseModel):
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", description="OWNER/REPO")
    run_id: int = Field(description="The Actions run's id")


class GithubFailedLog(Tool[GithubFailedLogInput]):
    name = "github_failed_log"
    description = "The log of the failed steps of a GitHub Actions run."
    input_model = GithubFailedLogInput
    default_policy = "allow"
    core = False

    def __init__(self, command: str | None = None) -> None:
        self._command = command

    def label(self, params: GithubFailedLogInput) -> str:
        return f"{params.repo} run {params.run_id}"

    async def run(self, params: GithubFailedLogInput, ctx: ToolContext) -> ToolResult:
        args = ["run", "view", str(params.run_id), "--repo", params.repo, "--log-failed"]
        return await run_gh(self._command or find_gh(), args)


def github_tools(command: str | None = None) -> list[Tool[Any]]:
    return [GithubGet(command), GithubApi(command), GithubFailedLog(command)]
