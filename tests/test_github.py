"""GitHub tools through a stand-in for the gh CLI."""

import json
import sys
from pathlib import Path

from phantasm.config import Config
from phantasm.core.cancellation import CancelToken
from phantasm.tools.base import ToolContext
from phantasm.tools.builtin.github import (
    GithubApi,
    GithubApiInput,
    GithubFailedLog,
    GithubFailedLogInput,
    GithubGet,
    GithubGetInput,
    run_gh,
)
from phantasm.tools.permissions import PathGuard

FAKE_GH = """#!{python}
import json, os, sys
args = sys.argv[1:]
body = sys.stdin.read() if "--input" in args else None
if "fail" in args[1]:
    sys.stderr.write("HTTP 404: Not Found\\n")
    sys.exit(1)
print(json.dumps({{"args": args, "body": body, "prompt": os.environ.get("GH_PROMPT_DISABLED")}}))
"""


def fake_gh(tmp_path: Path) -> str:
    path = tmp_path / "gh"
    path.write_text(FAKE_GH.format(python=sys.executable), encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def context() -> ToolContext:
    return ToolContext(
        session_id="s",
        turn_id="t",
        client_id=None,
        cancel=CancelToken(),
        config=Config(),
        paths=PathGuard.from_config(Config().permissions),
    )


def seen(result) -> dict:
    return json.loads(result.content[0].text)


async def test_get_reads_through_gh(tmp_path: Path) -> None:
    tool = GithubGet(fake_gh(tmp_path))
    result = await tool.run(GithubGetInput(path="/repos/a/b/issues", paginate=True), context())
    assert seen(result) == {
        "args": ["api", "repos/a/b/issues", "--method", "GET", "--paginate"],
        "body": None,
        "prompt": "1",
    }
    assert tool.default_policy == "allow" and tool.core is False


async def test_api_writes_with_a_body_and_asks(tmp_path: Path) -> None:
    tool = GithubApi(fake_gh(tmp_path))
    params = GithubApiInput(method="POST", path="repos/a/b/issues", body={"title": "Bug"})
    result = await tool.run(params, context())
    assert seen(result)["args"] == ["api", "repos/a/b/issues", "--method", "POST", "--input", "-"]
    assert json.loads(seen(result)["body"]) == {"title": "Bug"}
    assert tool.default_policy == "ask"
    assert tool.label(params) == "POST repos/a/b/issues"


async def test_failed_log_and_errors(tmp_path: Path) -> None:
    command = fake_gh(tmp_path)
    log = await GithubFailedLog(command).run(GithubFailedLogInput(repo="a/b", run_id=7), context())
    assert seen(log)["args"] == ["run", "view", "7", "--repo", "a/b", "--log-failed"]
    failed = await GithubGet(command).run(GithubGetInput(path="repos/fail"), context())
    assert failed.is_error and "HTTP 404" in failed.content[0].text  # type: ignore[union-attr]
    missing = await run_gh(None, ["api", "x"])
    assert missing.is_error and "not installed" in missing.content[0].text  # type: ignore[union-attr]
