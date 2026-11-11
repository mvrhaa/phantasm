"""Tests for the protocol models."""

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from zero.protocol.errors import PROTOCOL_VERSION, ErrorCode
from zero.protocol.messages import ContentPart, Envelope, FilePart, ImagePart, TextPart

content_part = TypeAdapter(ContentPart)


def test_text_part_is_parsed() -> None:
    part = content_part.validate_json('{ "type": "text", "text": "hello" }')

    assert part == TextPart(text="hello")


def test_image_part_is_parsed() -> None:
    part = content_part.validate_json(
        '{ "type": "image", "mime": "image/png", "data_b64": "aGVsbG8=" }'
    )

    assert part == ImagePart(mime="image/png", data_b64="aGVsbG8=")


def test_file_part_is_parsed() -> None:
    part = content_part.validate_json(
        '{ "type": "file", "name": "report.pdf", "path": "/tmp/report.pdf" }'
    )

    assert part == FilePart(name="report.pdf", path="/tmp/report.pdf")


@pytest.mark.parametrize(
    "part",
    [
        TextPart(text="hello"),
        ImagePart(mime="image/png", data_b64="aGVsbG8="),
        FilePart(name="report.pdf", path="/tmp/report.pdf"),
    ],
)
def test_content_part_survives_a_json_round_trip(part: BaseModel) -> None:
    dumped = part.model_dump_json()

    assert f'"type":"{part.type}"' in dumped
    assert content_part.validate_json(dumped) == part


@pytest.mark.parametrize(
    "data",
    [
        {"type": "video", "url": "http://example.com"},
        {"text": "no type"},
        {"type": "text"},
        {"type": "image", "mime": "image/png"},
        {"type": "file", "name": "report.pdf"},
        {"type": "text", "text": 42},
        "just a string",
    ],
    ids=[
        "unknown-type",
        "missing-type",
        "text-without-text",
        "image-without-data",
        "file-without-path",
        "text-not-a-string",
        "not-an-object",
    ],
)
def test_invalid_content_part_is_rejected(data: Any) -> None:
    with pytest.raises(ValidationError):
        content_part.validate_python(data)


def test_content_parts_work_as_a_list_field() -> None:
    class Holder(BaseModel):
        content: list[ContentPart]

    holder = Holder.model_validate(
        {
            "content": [
                {"type": "text", "text": "see the picture"},
                {"type": "image", "mime": "image/jpeg", "data_b64": "AAAA"},
                {"type": "file", "name": "a.txt", "path": "/tmp/a.txt"},
            ]
        }
    )

    assert [type(part) for part in holder.content] == [TextPart, ImagePart, FilePart]


def test_envelope_requires_only_type() -> None:
    envelope = Envelope.model_validate({"type": "ping"})

    assert envelope.type == "ping"
    assert envelope.id is None
    assert envelope.reply_to is None


def test_envelope_carries_id_and_reply_to() -> None:
    envelope = Envelope.model_validate({"type": "pong", "id": "a1", "reply_to": "b2"})

    assert envelope.id == "a1"
    assert envelope.reply_to == "b2"


def test_envelope_without_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Envelope.model_validate({"id": "a1"})


def test_error_codes_match_the_spec() -> None:
    assert {code.value for code in ErrorCode} == {
        "unauthorized",
        "protocol_mismatch",
        "unknown_type",
        "invalid_message",
        "session_not_found",
        "turn_busy",
        "provider_error",
        "provider_not_configured",
        "tool_error",
        "action_timeout",
        "client_gone",
        "internal",
    }


def test_error_code_is_a_plain_string_in_json() -> None:
    class Holder(BaseModel):
        code: ErrorCode

    assert ErrorCode.TURN_BUSY == "turn_busy"
    assert Holder(code=ErrorCode.TURN_BUSY).model_dump_json() == '{"code":"turn_busy"}'
    assert Holder.model_validate({"code": "turn_busy"}).code is ErrorCode.TURN_BUSY


def test_protocol_version_is_1() -> None:
    assert PROTOCOL_VERSION == 1
