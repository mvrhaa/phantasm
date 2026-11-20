"""Plugins the model writes: install, tools from @tool functions, failures, removal."""

import json
from pathlib import Path

import pytest

from phantasm.config import Config
from phantasm.core.cancellation import CancelToken
from phantasm.plugins import PluginHost
from phantasm.tools.base import ToolContext
from phantasm.tools.builtin.plugins import NoInput, PluginInstall, PluginList, PluginNameInput
from phantasm.tools.permissions import PathGuard

TOOLS = '''
from phantasm.plugin_api import tool


@tool
def shout(text: str, times: int = 1) -> str:
    """Shout the text. Loudly."""
    return " ".join([text.upper()] * times)


@tool(name="sum_up")
async def add(numbers: list[int]) -> dict:
    """Add numbers."""
    return {"sum": sum(numbers)}


@tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("no luck")


def helper():
    return 1
'''


def plugin(folder: Path, name: str, tools: str = TOOLS, manifest: dict | None = None) -> Path:
    path = folder / name
    path.mkdir(parents=True)
    data = (
        manifest if manifest is not None else {"name": name, "version": "1.0", "description": "x"}
    )
    (path / "plugin.json").write_text(json.dumps(data), encoding="utf-8")
    (path / "tools.py").write_text(tools, encoding="utf-8")
    return path


def context() -> ToolContext:
    return ToolContext(
        session_id="s",
        turn_id="t",
        client_id=None,
        cancel=CancelToken(),
        config=Config(),
        paths=PathGuard.from_config(Config().permissions),
    )


async def test_installed_plugin_gives_tools(tmp_path: Path) -> None:
    plugin(tmp_path, "loud")
    host = PluginHost(tmp_path)
    tools = {tool.name: tool for tool in host.install("loud")}
    assert sorted(tools) == ["broken", "shout", "sum_up"]
    shout = tools["shout"]
    assert shout.core is False and shout.default_policy == "ask"
    assert shout.description == "Shout the text. Loudly."
    assert shout.spec().input_schema["required"] == ["text"]
    params = shout.input_model.model_validate({"text": "hi", "times": 2})
    assert (await shout.run(params, context())).content[0].text == "HI HI"  # type: ignore[union-attr]
    total = tools["sum_up"].input_model.model_validate({"numbers": [1, 2]})
    assert (await tools["sum_up"].run(total, context())).content[0].text == '{"sum": 3}'  # type: ignore[union-attr]
    failed = await tools["broken"].run(tools["broken"].input_model(), context())
    assert failed.is_error and "RuntimeError: no luck" in failed.content[0].text  # type: ignore[union-attr]


def test_installed_plugins_load_again_and_can_be_removed(tmp_path: Path) -> None:
    plugin(tmp_path, "loud")
    PluginHost(tmp_path).install("loud")
    again = PluginHost(tmp_path)
    again.load_installed()
    assert len(again.tools()) == 3
    assert again.remove("loud") and again.tools() == []
    assert PluginHost(tmp_path).installed() == []
    assert not again.remove("loud")


@pytest.mark.parametrize(
    ("name", "tools", "manifest", "problem"),
    [
        ("bad", "import nothing_here\n", None, "ModuleNotFoundError"),
        ("empty", "x = 1\n", None, "no function marked"),
        ("named", TOOLS, {"name": "other"}, '"name": "named"'),
        ("Bad-Name", TOOLS, None, "lowercase"),
    ],
)
def test_broken_plugins_say_what_is_wrong(
    tmp_path: Path, name: str, tools: str, manifest: dict | None, problem: str
) -> None:
    plugin(tmp_path, name, tools, manifest)
    host = PluginHost(tmp_path)
    with pytest.raises(ValueError, match=problem):
        host.install(name)
    assert host.installed() == []


async def test_install_tool_always_asks_and_lists(tmp_path: Path) -> None:
    plugin(tmp_path, "loud")
    host = PluginHost(tmp_path)
    install = PluginInstall(host)
    preview = await install.preview(PluginNameInput(name="loud"), context())
    assert preview.approval_reason is not None
    done = await install.run(PluginNameInput(name="loud"), context())
    assert "Installed loud: broken, shout, sum_up" in done.content[0].text  # type: ignore[union-attr]
    listed = await PluginList(host).run(NoInput(), context())
    assert "- loud 1.0 (installed): x [broken, shout, sum_up]" in listed.content[0].text  # type: ignore[union-attr]
    assert str(tmp_path) in install.description
