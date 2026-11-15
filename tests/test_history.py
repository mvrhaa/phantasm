"""Tests for the history window in zero.core.context."""

from datetime import UTC, datetime
from typing import Any

from zero.core.context import IMAGE_OMITTED, build_history, clear_old_results, estimate_tokens
from zero.protocol.messages import ContentPart, FilePart, ImagePart, TextPart
from zero.providers.base import Message, ToolCall
from zero.store.messages import StoredMessage

WHEN = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CALL = {"id": "c1", "name": "read_file", "input": {"path": "a.txt"}}


def stored(role: str, text: str = "", **extra: Any) -> StoredMessage:
    content: list[ContentPart] = extra.pop("content", None) or (
        [TextPart(text=text)] if text else []
    )
    return StoredMessage(
        id=f"m{id(content)}", session_id="s1", role=role, content=content, created_at=WHEN, **extra
    )


def user(text: str, **extra: Any) -> StoredMessage:
    return stored("user", text, **extra)


def assistant(text: str = "", **extra: Any) -> StoredMessage:
    return stored("assistant", text, **extra)


def tool_result(text: str, call_id: str = "c1", **extra: Any) -> StoredMessage:
    return stored("tool", text, tool_call_id=call_id, **extra)


def one_token_each(message: Message) -> int:
    return 1


def texts(messages: list[Message]) -> list[str]:
    return [
        "".join(part.text for part in message.content if isinstance(part, TextPart))
        for message in messages
    ]


def test_everything_fits_into_a_large_budget() -> None:
    history = [user("one"), assistant("two"), user("three"), assistant("four")]

    result = build_history(history, budget=1000)

    assert texts(result) == ["one", "two", "three", "four"]
    assert [message.role for message in result] == ["user", "assistant", "user", "assistant"]


def test_empty_history() -> None:
    assert build_history([], budget=1000) == []


def test_oldest_turns_are_dropped_first() -> None:
    history = [
        user("turn 1"),
        assistant("reply 1"),
        user("turn 2"),
        assistant("reply 2"),
        user("turn 3"),
        assistant("reply 3"),
    ]

    result = build_history(history, budget=4, count_fn=one_token_each)

    assert texts(result) == ["turn 2", "reply 2", "turn 3", "reply 3"]


def test_history_is_cut_only_between_turns() -> None:
    history = [
        user("turn 1"),
        assistant("reply 1"),
        user("turn 2"),
        assistant("reply 2"),
    ]

    result = build_history(history, budget=3, count_fn=one_token_each)

    assert texts(result) == ["turn 2", "reply 2"]


def test_tool_call_and_its_result_are_never_separated() -> None:
    history = [
        user("old question"),
        assistant(tool_calls=[CALL]),
        tool_result("file content"),
        assistant("old answer"),
        user("new question"),
        assistant("new answer"),
    ]

    for budget in range(0, 8):
        result = build_history(history, budget=budget, count_fn=one_token_each)

        roles = [message.role for message in result]
        has_call = any(message.tool_calls for message in result)
        has_result = "tool" in roles
        assert has_call == has_result, f"pair split at budget {budget}: {roles}"
        assert result[0].role == "user", f"history starts mid-turn at budget {budget}"


def test_whole_turn_with_tools_is_kept_or_dropped_together() -> None:
    history = [
        user("old question"),
        assistant(tool_calls=[CALL]),
        tool_result("file content"),
        assistant("old answer"),
        user("new question"),
    ]

    fits = build_history(history, budget=5, count_fn=one_token_each)
    does_not_fit = build_history(history, budget=4, count_fn=one_token_each)

    assert len(fits) == 5
    assert texts(does_not_fit) == ["new question"]


def test_latest_turn_is_kept_even_over_budget() -> None:
    history = [user("old"), assistant("old reply"), user("x" * 10_000)]

    result = build_history(history, budget=10)

    assert texts(result) == ["x" * 10_000]


def test_stored_fields_are_converted_to_provider_messages() -> None:
    history = [
        user("read it"),
        assistant("looking", tool_calls=[CALL]),
        tool_result("no such file", is_error=True),
    ]

    _, with_call, result = build_history(history, budget=1000)

    assert with_call.tool_calls == [ToolCall(id="c1", name="read_file", input={"path": "a.txt"})]
    assert (result.role, result.tool_call_id, result.is_error) == ("tool", "c1", True)


def test_images_from_earlier_turns_are_replaced_with_a_note() -> None:
    image = ImagePart(mime="image/png", data_b64="AAAA")
    history = [
        user("", content=[TextPart(text="look at this"), image]),
        assistant("nice"),
        user("and now?"),
    ]

    result = build_history(history, budget=10_000)

    assert result[0].content == [TextPart(text="look at this"), TextPart(text=IMAGE_OMITTED)]
    assert IMAGE_OMITTED == "[image omitted]"


def test_images_in_the_latest_turn_are_kept() -> None:
    image = ImagePart(mime="image/png", data_b64="AAAA")
    history = [
        user("old"),
        assistant("old reply"),
        user("take a screenshot"),
        assistant(tool_calls=[{"id": "c1", "name": "screenshot", "input": {}}]),
        tool_result("", content=[image]),
    ]

    result = build_history(history, budget=10_000)

    assert result[-1].content == [image]


def test_files_are_kept_in_every_turn() -> None:
    file = FilePart(name="a.txt", path="/tmp/a.txt")
    history = [user("", content=[file]), assistant("got it"), user("thanks")]

    assert build_history(history, budget=10_000)[0].content == [file]


def test_stored_messages_are_not_modified() -> None:
    image = ImagePart(mime="image/png", data_b64="AAAA")
    first = user("", content=[image])

    build_history([first, assistant("ok"), user("next")], budget=10_000)

    assert first.content == [image]


def test_messages_before_the_first_user_message_form_a_turn() -> None:
    history = [assistant("scheduled note"), user("hi"), assistant("hello")]

    assert texts(build_history(history, budget=1000)) == ["scheduled note", "hi", "hello"]
    assert texts(build_history(history, budget=2, count_fn=one_token_each)) == ["hi", "hello"]


def test_estimate_uses_3_5_characters_per_token() -> None:
    assert estimate_tokens(Message(role="user", content=[TextPart(text="x" * 35)])) == 10
    assert estimate_tokens(Message(role="user", content=[TextPart(text="x" * 36)])) == 11
    assert estimate_tokens(Message(role="user")) == 0


def test_estimate_counts_images_and_tool_calls() -> None:
    image_message = Message(role="user", content=[ImagePart(mime="image/png", data_b64="A" * 9000)])
    call_message = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read_file")])

    assert estimate_tokens(image_message) == 1000
    assert estimate_tokens(call_message) > 0


def test_default_estimate_drives_the_window() -> None:
    history = [user("a" * 350), assistant("b" * 350), user("c" * 35)]

    assert texts(build_history(history, budget=210)) == ["a" * 350, "b" * 350, "c" * 35]
    assert texts(build_history(history, budget=209)) == ["c" * 35]


def tool_turn(question: str, result: str, call_id: str, **extra: Any) -> list[StoredMessage]:
    call = {"id": call_id, "name": "run_command", "input": {"command": "ls"}}
    return [
        user(question),
        assistant(tool_calls=[call]),
        tool_result(result, call_id=call_id, **extra),
        assistant("done"),
    ]


def cleared_flags(messages: list[StoredMessage]) -> list[bool]:
    return [
        message.content[0].text.startswith("[old result of")  # type: ignore[union-attr]
        for message in messages
        if message.role == "tool"
    ]


BIG = "x" * 3500  # about 1000 tokens


def test_old_tool_results_are_cleared_oldest_first_past_half_the_budget() -> None:
    history = [*tool_turn("a", BIG, "c1"), *tool_turn("b", BIG, "c2"), *tool_turn("c", BIG, "c3")]

    assert cleared_flags(clear_old_results(history, budget=4000)) == [True, True, False]
    assert cleared_flags(clear_old_results(history, budget=6000)) == [True, False, False]
    note = clear_old_results(history, budget=4000)[2].content[0]
    assert isinstance(note, TextPart)
    assert "run_command" in note.text


def test_nothing_is_cleared_under_half_the_budget() -> None:
    history = [*tool_turn("a", BIG, "c1"), *tool_turn("b", BIG, "c2")]
    assert clear_old_results(history, budget=100_000) == history


def test_errors_short_results_and_the_latest_turn_stay() -> None:
    history = [
        *tool_turn("a", BIG, "c1", is_error=True),
        *tool_turn("b", "ok", "c2"),
        *tool_turn("c", BIG * 3, "c3"),
    ]
    assert cleared_flags(clear_old_results(history, budget=1000)) == [False, False, False]


def test_clearing_keeps_ids_and_the_original_list() -> None:
    history = [*tool_turn("a", BIG, "c1"), *tool_turn("b", BIG, "c2")]
    cleared = clear_old_results(history, budget=2000)
    assert [message.id for message in cleared] == [message.id for message in history]
    assert history[2].content[0] == TextPart(text=BIG)
