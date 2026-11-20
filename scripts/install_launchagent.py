"""Install Phantasm as a LaunchAgent, so it starts by itself when you log in.

    uv run python scripts/install_launchagent.py              install (or reinstall) and start
    uv run python scripts/install_launchagent.py --print      show the plist, change nothing
    uv run python scripts/install_launchagent.py --uninstall  stop and remove

The agent runs this project's Python with ``-m phantasm``. launchd restarts it only
if it crashes: a clean exit (for example because Phantasm already runs) stays an
exit. Its output goes to ~/Library/Logs/Phantasm; Phantasm's own log is there too.
"""

import argparse
import os
import plistlib
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_LABEL = "local.phantasm.brain"

Run = Callable[[Sequence[str]], int]


def plist_path(label: str, home: Path) -> Path:
    """Where the agent's plist lives."""
    return home / "Library" / "LaunchAgents" / f"{label}.plist"


def build_plist(label: str, python: str, logs: Path, path_env: str) -> dict[str, Any]:
    """The agent's description for launchd."""
    return {
        "Label": label,
        "ProgramArguments": [python, "-m", "phantasm"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",
        "EnvironmentVariables": {"PATH": path_env},
        "StandardOutPath": str(logs / "launchd.out.log"),
        "StandardErrorPath": str(logs / "launchd.err.log"),
    }


def _launchctl(run: Run, *args: str) -> int:
    return run(["launchctl", *args])


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def install(label: str, home: Path, python: str, path_env: str, uid: int, run: Run = _run) -> Path:
    """Write the plist and (re)load it into the user's launchd domain; returns the plist."""
    logs = home / "Library" / "Logs" / "Phantasm"
    logs.mkdir(parents=True, exist_ok=True)
    target = plist_path(label, home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(plistlib.dumps(build_plist(label, python, logs, path_env)))
    _launchctl(run, "bootout", f"gui/{uid}/{label}")
    if _launchctl(run, "bootstrap", f"gui/{uid}", str(target)) != 0:
        raise RuntimeError(f"launchctl could not load {target}")
    return target


def uninstall(label: str, home: Path, uid: int, run: Run = _run) -> bool:
    """Stop the agent and remove its plist. False if it was not installed."""
    target = plist_path(label, home)
    _launchctl(run, "bootout", f"gui/{uid}/{label}")
    if not target.exists():
        return False
    target.unlink()
    return True


def main(argv: Sequence[str] | None = None, run: Run = _run) -> int:
    parser = argparse.ArgumentParser(description="Start Phantasm by itself at login.")
    parser.add_argument("--uninstall", action="store_true", help="stop and remove the agent")
    parser.add_argument("--print", action="store_true", help="show the plist, change nothing")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="launchd label of the agent")
    arguments = parser.parse_args(argv)
    home = Path.home()
    uid = os.getuid()

    if arguments.uninstall:
        removed = uninstall(arguments.label, home, uid, run)
        print("Removed" if removed else "Not installed:", plist_path(arguments.label, home))
        return 0
    path_env = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    if arguments.print:
        logs = home / "Library" / "Logs" / "Phantasm"
        plist = build_plist(arguments.label, sys.executable, logs, path_env)
        print(plistlib.dumps(plist).decode())
        return 0
    try:
        target = install(arguments.label, home, sys.executable, path_env, uid, run)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Installed {target}; Phantasm now starts at login. Check: curl 127.0.0.1:47400/health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
