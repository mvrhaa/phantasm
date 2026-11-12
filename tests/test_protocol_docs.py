"""Tests of scripts/gen_protocol_docs.py and the reference it writes."""

import importlib.util
from pathlib import Path
from types import ModuleType

from pydantic import TypeAdapter

from zero import __version__
from zero.protocol.messages import ClientMessage, ServerEvent

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "gen_protocol_docs.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_protocol_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_in_the_repository_is_up_to_date() -> None:
    assert (ROOT / "docs" / "protocol.md").read_text(encoding="utf-8") == load().render(), (
        "run: uv run python scripts/gen_protocol_docs.py"
    )


def test_every_message_has_a_section_and_its_example_is_valid() -> None:
    gen = load()
    text = gen.render()

    for union, examples in (
        (ClientMessage, gen.CLIENT_EXAMPLES),
        (ServerEvent, gen.SERVER_EXAMPLES),
    ):
        adapter = TypeAdapter(union)
        names = [gen.type_of(model) for model in gen.members(union)]
        assert sorted(names) == sorted(gen.type_of(type(e)) for e in examples)
        for example in examples:
            data = example.model_dump(mode="json", exclude_none=True)
            assert adapter.validate_python(data) == example
        for name in names:
            assert f"### `{name}`" in text


def test_the_reference_names_the_version() -> None:
    assert f"Zero {__version__}." in load().render()
    assert __version__ == "0.1.0"
