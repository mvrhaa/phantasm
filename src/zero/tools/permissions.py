"""The sandbox: which paths tools may touch."""

import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from zero.config import PermissionsConfig

_RELATIVE_HINT = "use an absolute path or one that starts with ~"


class PathDenied(Exception):
    """A tool asked for a path outside the sandbox. The message is written for the model."""


def _key(path: Path) -> tuple[str, ...]:
    """The parts of a path in a form that ignores letter case and Unicode composition.

    The default macOS file system treats ``.SSH`` and ``.ssh`` (or a composed
    and a decomposed ``é``) as the same name, and ``Path.resolve()`` keeps
    whatever spelling it was given, so paths are compared by these keys.
    """
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in path.parts)


def _inside(path: tuple[str, ...], folder: tuple[str, ...]) -> bool:
    return path[: len(folder)] == folder


def _shown(path: Path) -> str:
    """The path as the user would write it, with the home folder as ``~``."""
    home = Path.home().resolve()
    if path == home:
        return "~"
    if path.is_relative_to(home):
        return f"~/{path.relative_to(home)}"
    return str(path)


def _resolve_setting(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"sandbox path {str(raw)!r} must be absolute or start with ~")
    return path.resolve()


class PathGuard:
    """Decides whether a tool may use a path.

    A path is allowed when, once ``~`` is expanded and symlinks are followed,
    it lies inside one of ``allowed_roots`` and outside every entry of
    ``denied_paths``. A denied entry wins over a root that contains it. Letter
    case is ignored, as on the default macOS file system, so ``~/.SSH`` does
    not slip past a ban on ``~/.ssh``. Relative paths are refused.

    Raises ``ValueError`` if a root or a denied entry is itself relative.
    """

    def __init__(self, allowed_roots: Iterable[str | Path], denied_paths: Iterable[str | Path]):
        self.allowed_roots = [_resolve_setting(root) for root in allowed_roots]
        self.denied_paths = [_resolve_setting(path) for path in denied_paths]
        self._allowed = [_key(root) for root in self.allowed_roots]
        self._denied = [_key(path) for path in self.denied_paths]

    @classmethod
    def from_config(cls, permissions: PermissionsConfig) -> Self:
        """The guard described by the ``permissions`` section of the config."""
        return cls(permissions.allowed_roots, permissions.denied_paths)

    def check(self, raw: str | Path) -> Path:
        """Return the resolved path if tools may use it, otherwise raise ``PathDenied``.

        The path does not have to exist, so a file can be checked before it
        is created. Following symlinks touches the file system, so call this
        where a short blocking call is fine, e.g. inside ``asyncio.to_thread``.
        """
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise PathDenied(f"{str(raw)!r} is a relative path; {_RELATIVE_HINT}.")
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            raise PathDenied(f"{str(raw)!r} cannot be resolved to a real path.") from None

        key = _key(resolved)
        if any(_inside(key, denied) for denied in self._denied):
            raise PathDenied(f"{_shown(resolved)} is in a protected location.")
        if not any(_inside(key, root) for root in self._allowed):
            roots = ", ".join(_shown(root) for root in self.allowed_roots) or "none"
            raise PathDenied(f"{_shown(resolved)} is outside the folders tools may use ({roots}).")
        return resolved
