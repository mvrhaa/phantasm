"""Tests for phantasm.tools.permissions.PathGuard: the sandbox for paths."""

import os
from pathlib import Path

import pytest

from phantasm.config import Config
from phantasm.tools.permissions import PathDenied, PathGuard


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home folder with a secret key and a note; ``~`` points at it."""
    home = tmp_path.resolve() / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("secret")
    (home / "notes.md").write_text("hi")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def outside(home: Path) -> Path:
    """A folder next to the home folder, outside the sandbox."""
    folder = home.parent / "outside"
    folder.mkdir()
    (folder / "secret.txt").write_text("secret")
    return folder


@pytest.fixture
def guard(home: Path) -> PathGuard:
    return PathGuard(["~"], ["~/.ssh"])


def denial(guard: PathGuard, path: str | Path) -> str:
    with pytest.raises(PathDenied) as error:
        guard.check(path)
    return str(error.value)


def test_path_inside_a_root_is_returned_resolved(guard: PathGuard, home: Path) -> None:
    assert guard.check("~/notes.md") == home / "notes.md"
    assert guard.check(home / "notes.md") == home / "notes.md"
    assert guard.check(str(home / "notes.md")) == home / "notes.md"


def test_root_itself_is_allowed(guard: PathGuard, home: Path) -> None:
    assert guard.check("~") == home


def test_missing_file_inside_a_root_is_allowed(guard: PathGuard, home: Path) -> None:
    assert guard.check("~/new/folder/file.txt") == home / "new" / "folder" / "file.txt"


def test_dot_dot_that_stays_inside_is_allowed(guard: PathGuard, home: Path) -> None:
    assert guard.check("~/a/b/../../notes.md") == home / "notes.md"


def test_name_that_only_starts_like_a_protected_one_is_allowed(
    guard: PathGuard, home: Path
) -> None:
    assert guard.check("~/.sshx/config") == home / ".sshx" / "config"


def test_symlinked_root_can_be_reached_through_either_path(tmp_path: Path, home: Path) -> None:
    real = tmp_path.resolve() / "real"
    real.mkdir()
    link = tmp_path.resolve() / "link"
    link.symlink_to(real)
    guard = PathGuard([link], [])

    assert guard.check(link / "file.txt") == real / "file.txt"
    assert guard.check(real / "file.txt") == real / "file.txt"


def test_case_is_ignored_for_roots(tmp_path: Path, home: Path) -> None:
    guard = PathGuard([tmp_path.resolve() / "Projects"], [])

    assert guard.check(tmp_path.resolve() / "projects" / "app.py")


def test_path_outside_the_roots_is_denied(guard: PathGuard, outside: Path) -> None:
    assert denial(guard, outside / "secret.txt") == (
        f"{outside / 'secret.txt'} is outside the folders tools may use (~)."
    )


def test_dot_dot_cannot_climb_out_of_a_root(guard: PathGuard, home: Path, outside: Path) -> None:
    assert "outside the folders" in denial(guard, "~/../outside/secret.txt")
    assert "outside the folders" in denial(guard, f"{home}/notes/../../outside")


def test_folder_sharing_a_prefix_with_a_root_is_outside(tmp_path: Path, home: Path) -> None:
    guard = PathGuard([tmp_path.resolve() / "proj"], [])

    assert "outside the folders" in denial(guard, tmp_path.resolve() / "project2" / "x")


def test_symlink_from_a_root_to_outside_is_denied(
    guard: PathGuard, home: Path, outside: Path
) -> None:
    (home / "shortcut").symlink_to(outside)
    (home / "secret-link.txt").symlink_to(outside / "secret.txt")

    assert "outside the folders" in denial(guard, "~/shortcut/secret.txt")
    assert "outside the folders" in denial(guard, "~/shortcut")
    assert "outside the folders" in denial(guard, "~/secret-link.txt")


def test_no_roots_allow_nothing(home: Path) -> None:
    assert denial(PathGuard([], []), "~/notes.md") == (
        "~/notes.md is outside the folders tools may use (none)."
    )


@pytest.mark.parametrize("path", ["~/.ssh", "~/.ssh/id_rsa", "~/.ssh/new_key"])
def test_protected_folder_and_everything_in_it_are_denied(guard: PathGuard, path: str) -> None:
    assert denial(guard, path).endswith("is in a protected location.")


def test_message_shows_the_home_folder_as_tilde(guard: PathGuard) -> None:
    assert denial(guard, "~/.ssh/id_rsa") == "~/.ssh/id_rsa is in a protected location."


def test_symlink_into_a_protected_folder_is_denied(guard: PathGuard, home: Path) -> None:
    (home / "keys").symlink_to(home / ".ssh")

    assert "protected location" in denial(guard, "~/keys/id_rsa")


@pytest.mark.parametrize("path", ["~/.SSH/id_rsa", "~/.Ssh", "~/.sSh/ID_RSA"])
def test_letter_case_cannot_slip_past_a_protected_folder(guard: PathGuard, path: str) -> None:
    assert "protected location" in denial(guard, path)


def test_unicode_composition_cannot_slip_past_a_protected_folder(home: Path) -> None:
    guard = PathGuard(["~"], ["~/café"])

    assert "protected location" in denial(guard, "~/café/menu.txt")


def test_protected_entry_wins_over_a_root(home: Path) -> None:
    guard = PathGuard(["~/.ssh"], ["~/.ssh"])

    assert "protected location" in denial(guard, "~/.ssh/id_rsa")


def test_default_config_protects_ssh_and_keychains(home: Path) -> None:
    guard = PathGuard.from_config(Config().permissions)

    assert guard.check("~/notes.md") == home / "notes.md"
    assert "protected location" in denial(guard, "~/.ssh/id_rsa")
    assert "protected location" in denial(guard, "~/Library/Keychains/login.keychain-db")


@pytest.mark.parametrize("path", ["notes.md", "./notes.md", "../outside", ""])
def test_relative_paths_are_refused(guard: PathGuard, path: str) -> None:
    assert denial(guard, path) == (
        f"{path!r} is a relative path; use an absolute path or one that starts with ~."
    )


def test_path_with_a_null_byte_is_refused(guard: PathGuard) -> None:
    assert "cannot be resolved" in denial(guard, "~/notes\x00.md")


def test_symlink_loop_is_refused(guard: PathGuard, home: Path) -> None:
    loop = home / "loop"
    os.symlink(loop, loop)

    assert "cannot be resolved" in denial(guard, "~/loop/file")


@pytest.mark.parametrize(
    ("roots", "denied"), [(["projects"], []), (["~"], ["secrets"]), (["./here"], [])]
)
def test_relative_settings_are_rejected(roots: list[str], denied: list[str]) -> None:
    with pytest.raises(ValueError, match="must be absolute or start with ~"):
        PathGuard(roots, denied)
