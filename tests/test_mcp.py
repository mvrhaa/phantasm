"""MCP servers from mcp.json as tools, against ``tests/fake_mcp.py``."""

import json
import sys
from pathlib import Path

import pytest

from tests.test_turn_text import connected, running, send, until_finished
from zero.core.cancellation import CancelToken
from zero.mcp_client import McpClient, McpError
from zero.paths import home_dir
from zero.protocol.messages import ImagePart, TextPart
from zero.providers.base import TextDelta
from zero.providers.fake import FakeProvider
from zero.tools.mcp import McpHost, to_result, tool_name

FAKE = str(Path(__file__).with_name("fake_mcp.py"))


def write_config(path: Path, servers: dict) -> Path:
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return path


def fake_server(**env: str) -> dict:
    return {"command": sys.executable, "args": [FAKE], "env": env}


async def test_client_lists_and_calls_tools() -> None:
    client = McpClient("fake", sys.executable, [FAKE])
    await client.start()
    try:
        tools = await client.list_tools()
        assert [tool["name"] for tool in tools] == ["echo", "picture", "fail", "die"]
        result = await client.call("echo", {"text": "hi"}, timeout=10)
        assert result["content"][0]["text"] == "hi roots=[]"
        with pytest.raises(McpError, match="unknown"):
            await client.request("nope/nothing", {})
    finally:
        await client.close()


async def test_client_reports_a_command_that_cannot_start(tmp_path: Path) -> None:
    client = McpClient("ghost", str(tmp_path / "missing"))
    with pytest.raises(McpError, match="cannot start"):
        await client.start()


async def test_host_makes_tools_and_fills_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAG_FROM_ENV", "tagged")
    config = write_config(
        tmp_path / "mcp.json", {"fake": fake_server(FAKE_MCP_TAG="${TAG_FROM_ENV}")}
    )
    host = McpHost(config)
    await host.start()
    try:
        tools = {tool.name: tool for tool in host.tools()}
        assert sorted(tools) == [
            "mcp__fake__die",
            "mcp__fake__echo",
            "mcp__fake__fail",
            "mcp__fake__picture",
        ]
        echo = tools["mcp__fake__echo"]
        assert echo.core is False and echo.default_policy == "ask"
        assert echo.spec().input_schema["required"] == ["text"]
        result = await host.call("fake", "echo", {"text": "hi"}, CancelToken())
        assert result["content"][0]["text"] == "hi tagged roots=[]"
    finally:
        await host.close()


async def test_a_stopped_server_starts_again_on_its_next_call(tmp_path: Path) -> None:
    host = McpHost(write_config(tmp_path / "mcp.json", {"fake": fake_server()}))
    await host.start()
    try:
        with pytest.raises(McpError):
            await host.call("fake", "die", {}, CancelToken())
        result = await host.call("fake", "echo", {"text": "back"}, CancelToken())
        assert result["content"][0]["text"].startswith("back")
    finally:
        await host.close()


async def test_bad_servers_are_noted_and_skipped(tmp_path: Path) -> None:
    config = write_config(
        tmp_path / "mcp.json",
        {
            "broken": {"args": []},
            "ghost": {"command": str(tmp_path / "none")},
            "fake": fake_server(),
        },
    )
    host = McpHost(config)
    await host.start()
    try:
        assert set(host.problems) == {"broken", "ghost"}
        assert all(tool.name.startswith("mcp__fake__") for tool in host.tools())
    finally:
        await host.close()
    assert McpHost(tmp_path / "absent.json").read_config() == {}


def test_results_become_text_and_images() -> None:
    result = to_result(
        {
            "content": [
                {"type": "image", "data": "abc", "mimeType": "image/jpeg"},
                {"type": "resource", "resource": {"uri": "file:///x", "text": "inside"}},
                {"type": "resource_link", "uri": "file:///y"},
            ],
            "isError": True,
        }
    )
    assert result.is_error
    assert result.content == [
        ImagePart(mime="image/jpeg", data_b64="abc"),
        TextPart(text="inside"),
        TextPart(text="[resource file:///y]"),
    ]
    assert to_result({"structuredContent": {"a": 1}}).content == [TextPart(text='{"a": 1}')]
    assert to_result({}).content == [TextPart(text="(no output)")]
    assert tool_name("my server", "do.it") == "mcp__my_server__do_it"


def test_mcp_tools_reach_the_turn_on_demand() -> None:
    write_config(home_dir() / "mcp.json", {"fake": fake_server()})
    provider = FakeProvider([[TextDelta(text="ok")], [TextDelta(text="title")]])
    with running(provider) as (client, state), connected(client, state) as ws:
        client.portal.call(state.mcp.start)
        send(ws, "hi")
        until_finished(ws)
    system = provider.calls[0].messages[0].content[0].text  # type: ignore[union-attr]
    assert "- mcp__fake__echo(text): Echo the text back" in system
    assert "mcp__fake__echo" not in {tool.name for tool in provider.calls[0].tools}
