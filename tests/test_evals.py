"""The evaluation runner: tasks, checks and a run through the real brain on a fake model."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from zero.cli import main
from zero.config import Config
from zero.evals import check_one, load_tasks, run_task, task_config
from zero.providers.base import TextDelta
from zero.providers.fake import FakeProvider


def test_every_task_loads_and_has_a_check() -> None:
    tasks = load_tasks()
    assert len(tasks) == 10
    assert all(task.get("check") for task in tasks)
    assert [t["name"] for t in load_tasks(["csv-to-json"])] == ["csv-to-json"]
    with pytest.raises(ValueError, match="no tasks named nope"):
        load_tasks(["nope"])


def test_task_config_keeps_the_model_and_grants_everything(tmp_path: Path) -> None:
    base = Config()
    base.provider.model = "my-model"
    config = task_config(base, tmp_path, 12)
    assert config.provider.model == "my-model"
    assert config.permissions.allowed_roots == [str(tmp_path)]
    assert config.permissions.bypass_approvals is True
    assert config.agent.max_steps == 12
    assert base.permissions.bypass_approvals is False


async def test_checks(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps([1, 2]))
    (tmp_path / "x.py").write_text("print('seven 7')\n")
    (tmp_path / "old.py").write_text("fetch_data()\n")
    state = SimpleNamespace(
        schedules=SimpleNamespace(list=_async([SimpleNamespace(task="Drink water")])),
        memory=SimpleNamespace(search=_async([])),
    )
    replies = ["The answer is Pixel."]
    assert await check_one({"command": "{python} x.py", "stdout": "7"}, tmp_path, [], state) is None
    failed = await check_one(
        {"command": "{python} -c 'import sys; sys.exit(2)'"}, tmp_path, [], state
    )
    assert failed is not None and "failed" in failed
    assert await check_one({"json_file": "a.json", "equals": [1, 2]}, tmp_path, [], state) is None
    assert await check_one({"json_file": "a.json", "equals": [2]}, tmp_path, [], state)
    assert await check_one({"file": "x.py", "contains": ["seven"]}, tmp_path, [], state) is None
    assert await check_one({"file": "nope.txt", "contains": []}, tmp_path, [], state)
    assert await check_one({"absent_in": "*.py", "text": "fetch_data"}, tmp_path, [], state)
    assert await check_one({"reply": ["pixel"]}, tmp_path, replies, state) is None
    assert await check_one({"reply": ["Mars"]}, tmp_path, replies, state)
    assert await check_one({"schedule": "water"}, tmp_path, [], state) is None
    assert await check_one({"memory": "Pixel"}, tmp_path, [], state)
    assert await check_one({"weird": 1}, tmp_path, [], state) == "unknown check {'weird': 1}"


def _async(value):
    async def answer(*args, **kwargs):
        return value

    return answer


def test_a_task_runs_through_the_brain_on_a_fake_model() -> None:
    home = os.environ["ZERO_HOME"]
    task = {"name": "t", "prompt": "Sign it.", "check": {"reply": ["Signed, Zero"]}}
    good = run_task(task, Config(), FakeProvider([[TextDelta(text="Signed, Zero")]] * 2))
    assert good.passed and good.detail == "ok"
    bad = run_task(task, Config(), FakeProvider([[TextDelta(text="No.")]] * 2))
    assert not bad.passed and "lack ['Signed, Zero']" in bad.detail
    assert os.environ["ZERO_HOME"] == home


def test_cli_refuses_unknown_tasks(capsys) -> None:
    assert main(["eval", "nope"]) == 1
    assert "no tasks named nope" in capsys.readouterr().out
