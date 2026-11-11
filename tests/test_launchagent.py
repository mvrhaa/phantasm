"""Tests of scripts/install_launchagent.py. launchctl is never really called."""

import importlib.util
import plistlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "install_launchagent.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("install_launchagent", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load()


class FakeLaunchctl:
    def __init__(self, fail_bootstrap: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.fail_bootstrap = fail_bootstrap

    def __call__(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        return 1 if self.fail_bootstrap and command[1] == "bootstrap" else 0


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_plist_restarts_only_after_a_crash(home: Path) -> None:
    plist = agent.build_plist("local.zero.brain", "/venv/bin/python", home / "logs", "/a:/b")

    assert plist["ProgramArguments"] == ["/venv/bin/python", "-m", "zero"]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["EnvironmentVariables"] == {"PATH": "/a:/b"}
    assert plist["StandardErrorPath"] == str(home / "logs" / "launchd.err.log")


def test_install_writes_the_plist_and_reloads_the_agent(home: Path) -> None:
    launchctl = FakeLaunchctl()

    target = agent.install("local.zero.brain", home, "/venv/bin/python", "/a", 501, launchctl)

    assert target == home / "Library" / "LaunchAgents" / "local.zero.brain.plist"
    written = plistlib.loads(target.read_bytes())
    assert written["Label"] == "local.zero.brain"
    assert (home / "Library" / "Logs" / "Zero").is_dir()
    assert launchctl.commands == [
        ["launchctl", "bootout", "gui/501/local.zero.brain"],
        ["launchctl", "bootstrap", "gui/501", str(target)],
    ]


def test_failed_load_is_reported(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = agent.main([], run=FakeLaunchctl(fail_bootstrap=True))

    assert code == 1
    assert "could not load" in capsys.readouterr().err


def test_uninstall_stops_and_removes_the_agent(home: Path) -> None:
    launchctl = FakeLaunchctl()
    target = agent.install("local.zero.brain", home, "/py", "/a", 501, launchctl)

    assert agent.uninstall("local.zero.brain", home, 501, launchctl)
    assert not target.exists()
    assert launchctl.commands[-1] == ["launchctl", "bootout", "gui/501/local.zero.brain"]
    assert not agent.uninstall("local.zero.brain", home, 501, launchctl)


def test_print_shows_the_plist_and_changes_nothing(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    launchctl = FakeLaunchctl()

    assert agent.main(["--print", "--label", "test.zero"], run=launchctl) == 0

    shown = plistlib.loads(capsys.readouterr().out.encode())
    assert shown["Label"] == "test.zero"
    assert launchctl.commands == []
    assert not (home / "Library").exists()


def test_main_installs_and_uninstalls(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    launchctl = FakeLaunchctl()

    assert agent.main([], run=launchctl) == 0
    assert "Zero now starts at login" in capsys.readouterr().out
    assert agent.main(["--uninstall"], run=launchctl) == 0
    assert capsys.readouterr().out.startswith("Removed")
