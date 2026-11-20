"""Evaluation tasks: real work with an outcome a script checks, run on the configured model.

    phantasm eval                          every task
    phantasm eval fix-failing-test --report results.json

Each task gets a throw-away data folder and working folder and runs through the
real brain (the WebSocket API), with the user's provider (or Claude Code, if it
thinks now) and every approval granted. It costs real model requests. A task is
a JSON file in ``eval_tasks``:

- ``prompt`` (or ``prompts``: a list of ``{"text", "new_session"}``) — what the user says;
  each message starts with the working folder's path.
- ``files`` — files of the working folder; ``home_files`` — files of the data folder
  (skills, plugins).
- ``check`` — one check or a list, all must pass: ``{"command", "stdout"?}`` exits 0
  (and prints ``stdout``); ``{"file", "contains"}``; ``{"absent_in", "text"}``;
  ``{"json_file", "equals"}``; ``{"reply": [words]}`` (all, any case);
  ``{"schedule": text}``; ``{"memory": text}``. ``{python}`` is this Python.
- ``max_steps``, ``timeout_s``.
"""

import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from phantasm import clock
from phantasm.config import Config

TASKS_DIR = Path(__file__).parent / "eval_tasks"
TIMEOUT_S = 900.0
COMMAND_TIMEOUT_S = 120


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: str
    seconds: float
    requests: int = 0
    tokens: int = 0


def load_tasks(names: list[str] | None = None) -> list[dict[str, Any]]:
    """The tasks, all or those named; an unknown name raises ``ValueError``."""
    tasks = [
        {**json.loads(path.read_text(encoding="utf-8")), "name": path.stem}
        for path in sorted(TASKS_DIR.glob("*.json"))
    ]
    if names:
        known = {task["name"] for task in tasks}
        missing = sorted(set(names) - known)
        if missing:
            raise ValueError(
                f"no tasks named {', '.join(missing)}; known: {', '.join(sorted(known))}"
            )
        tasks = [task for task in tasks if task["name"] in names]
    return tasks


@contextmanager
def _home(path: Path):
    before = os.environ.get("PHANTASM_HOME")
    os.environ["PHANTASM_HOME"] = str(path)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("PHANTASM_HOME", None)
        else:
            os.environ["PHANTASM_HOME"] = before


def _write(folder: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _prompts(task: dict[str, Any]) -> list[dict[str, Any]]:
    if "prompts" in task:
        return [p if isinstance(p, dict) else {"text": p} for p in task["prompts"]]
    return [{"text": task["prompt"]}]


def task_config(base: Config, work: Path, max_steps: int | None) -> Config:
    """The user's config for a task: their model, the working folder only, every approval given."""
    config = Config.model_validate(base.model_dump())
    config.permissions.allowed_roots = [str(work)]
    config.permissions.bypass_approvals = True
    config.agent.review_conversations = False
    if max_steps:
        config.agent.max_steps = max_steps
    return config


def run_task(task: dict[str, Any], base: Config, provider: Any = None) -> EvalResult:
    """Run one task through a brain of its own and check what it left."""
    from starlette.testclient import TestClient

    from phantasm.server.app import create_app
    from phantasm.server.state import get_state

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="phantasm-eval-") as temp:
        home, work = Path(temp) / "home", Path(temp) / "work"
        work.mkdir(parents=True)
        _write(work, task.get("files", {}))
        _write(home, task.get("home_files", {}))
        config = task_config(base, work, task.get("max_steps"))
        replies: list[str] = []
        with _home(home), TestClient(create_app(config, provider)) as client:
            state = get_state(client.app)  # type: ignore[arg-type]
            hello = {"type": "hello", "protocol": 1, "client": {"app": "eval", "client_id": "eval"}}
            with client.websocket_connect(f"/ws?token={state.token}") as ws:
                ws.send_json(hello)
                ws.receive_json()
                session: str | None = None
                deadline = time.monotonic() + task.get("timeout_s", TIMEOUT_S)
                for index, prompt in enumerate(_prompts(task)):
                    if prompt.get("new_session"):
                        session = None
                    text = f"[Working folder: {work}]\n{prompt['text']}"
                    message = {"type": "message.send", "id": f"m{index}", "text": text}
                    message["source"] = "desktop"
                    if session is not None:
                        message["session_id"] = session
                    ws.send_json(message)
                    finished = _until_finished(ws, deadline)
                    if finished is None:
                        return EvalResult(task["name"], False, "timed out", _since(started))
                    session = finished["session_id"]
                    replies.append(finished.get("text") or "")
                    replies += _quiet(client, state, ws, deadline)
            problem = client.portal.call(_check_all, task.get("check", []), work, replies, state)
            usage = client.portal.call(state.usage.totals, clock.now() - timedelta(days=1))
        return EvalResult(
            task["name"],
            problem is None,
            problem or "ok",
            _since(started),
            sum(total.requests for total in usage),
            sum(total.input_tokens + total.output_tokens for total in usage),
        )


def _since(started: float) -> float:
    return round(time.monotonic() - started, 1)


def _until_finished(ws: Any, deadline: float) -> dict[str, Any] | None:
    while time.monotonic() < deadline:
        event = ws.receive_json()
        if event.get("type") == "turn.finished":
            return event
        if event.get("type") == "error":
            return {**event, "text": f"error: {event.get('message')}", "session_id": None}
    return None


def _quiet(client: Any, state: Any, ws: Any, deadline: float) -> list[str]:
    """Wait while agents work and their reports start turns; the replies of those turns."""
    replies: list[str] = []
    while time.monotonic() < deadline:
        active = client.portal.call(state.agents.active)
        if not active and not state.turns.active:
            return replies
        time.sleep(0.5)
        while state.turns.active:
            finished = _until_finished(ws, deadline)
            if finished is None:
                return replies
            replies.append(finished.get("text") or "")
    return replies


async def _check_all(checks: Any, work: Path, replies: list[str], state: Any) -> str | None:
    for check in checks if isinstance(checks, list) else [checks]:
        problem = await check_one(check, work, replies, state)
        if problem is not None:
            return problem
    return None


async def check_one(
    check: dict[str, Any], work: Path, replies: list[str], state: Any
) -> str | None:
    """What is wrong with the outcome by one check, or None if it passes."""
    if "command" in check:
        command = check["command"].replace("{python}", shlex.quote(sys.executable))
        done = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
        if done.returncode != 0:
            return f"`{check['command']}` failed: {(done.stdout + done.stderr)[-400:]}"
        if "stdout" in check and str(check["stdout"]) not in done.stdout:
            return f"`{check['command']}` did not print {check['stdout']!r}: {done.stdout[-200:]}"
        return None
    if "file" in check:
        path = work / check["file"]
        if not path.is_file():
            return f"{check['file']} is missing"
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [part for part in check.get("contains", []) if part not in text]
        return f"{check['file']} lacks {missing}" if missing else None
    if "absent_in" in check:
        found = [str(p.relative_to(work)) for p in work.rglob(check["absent_in"]) if p.is_file()]
        still = [
            name for name in found if check["text"] in (work / name).read_text(errors="replace")
        ]
        return f"{check['text']!r} is still in {still}" if still else None
    if "json_file" in check:
        try:
            data = json.loads((work / check["json_file"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return f"{check['json_file']} is not readable JSON ({type(exc).__name__})"
        return None if data == check["equals"] else f"{check['json_file']} is {data!r}"
    if "reply" in check:
        said = "\n".join(replies).lower()
        missing = [word for word in check["reply"] if word.lower() not in said]
        return f"the replies lack {missing}: {said[-300:]!r}" if missing else None
    if "schedule" in check:
        schedules = await state.schedules.list(include_disabled=True)
        found = any(check["schedule"].lower() in s.task.lower() for s in schedules)
        return None if found else f"no schedule about {check['schedule']!r}"
    if "memory" in check:
        found = await state.memory.search(check["memory"], limit=5)
        return None if found else f"nothing remembered about {check['memory']!r}"
    return f"unknown check {check!r}"


def run(
    names: list[str] | None, base: Config, say: Callable[[str], None], report: Path | None = None
) -> list[EvalResult]:
    """Run the tasks one by one, print a line for each, write the report if asked."""
    results = []
    for task in load_tasks(names):
        say(f"… {task['name']}")
        result = run_task(task, base)
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        say(
            f"{mark} {result.name}  {result.seconds} s  {result.requests} requests  "
            f"{result.tokens} tokens  {result.detail}",
            flush=True,
        )
    passed = sum(result.passed for result in results)
    say(f"{passed}/{len(results)} passed")
    if report is not None:
        report.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return results
