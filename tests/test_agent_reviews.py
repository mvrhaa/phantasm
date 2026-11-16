"""Agents whose work a reviewer checks, progress notes, and waiting for reports in a turn."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from tests.agents import RunnerKit
from tests.test_agent_control import ended
from zero.agents.runner import REVIEWER, VERIFY_ROUNDS
from zero.core.cancellation import CancelToken
from zero.paths import db_path
from zero.store.agents import Agent
from zero.store.db import Database


@pytest.fixture
async def kit() -> AsyncIterator[RunnerKit]:
    async with Database(db_path()) as db:
        await db.migrate()
        kit = RunnerKit(db, max_agents=1)
        yield kit
        await kit.runner.close()
        await kit.turns.cancel_all(grace_s=1)


async def checked_agent(kit: RunnerKit) -> tuple[str, Agent]:
    parent = await kit.sessions.create()
    agent = await kit.runner.start("coder", "Make the tests pass", parent.id, verify=True)
    return parent.id, agent


async def reviewer_of(kit: RunnerKit, agent: Agent) -> Agent:
    assert agent.session_id is not None
    found = [a for a in await kit.agents.list(agent.session_id) if a.name == REVIEWER]
    return found[-1]


async def settle(kit: RunnerKit | None = None, notices: int = 0) -> None:
    """Let deliveries run: until ``kit`` has more than ``notices`` notices, or a short while."""
    for _ in range(200):
        await asyncio.sleep(0.005)
        if kit is not None and len(kit.notices) > notices:
            return


async def test_a_passing_review_sends_the_report_on(kit: RunnerKit) -> None:
    parent, agent = await checked_agent(kit)
    await kit.runner.finished(agent.id, ended("ok", "All tests pass."))
    reviewer = await reviewer_of(kit, agent)
    assert reviewer.review_of == agent.id
    assert kit.started[-1][0] == reviewer.id and "Make the tests pass" in kit.started[-1][1]
    assert "All tests pass." in kit.started[-1][1]
    assert (await kit.agents.get(agent.id)).status == "working"  # type: ignore[union-attr]

    await kit.runner.finished(reviewer.id, ended("ok", "PASS\nI ran pytest: 3 passed."))
    await settle(kit)
    done = await kit.agents.get(agent.id)
    assert done is not None and done.status == "done" and done.reviews == 1
    session, text = kit.notices[-1]
    assert session == parent
    assert "A reviewer checked its work: PASS" in text and "All tests pass." in text


async def test_a_failing_review_sends_the_agent_back_to_work(kit: RunnerKit) -> None:
    parent, agent = await checked_agent(kit)
    await kit.runner.finished(agent.id, ended("ok", "Done."))
    first = await reviewer_of(kit, agent)
    await kit.runner.finished(first.id, ended("ok", "FAIL\ntest_discount still fails"))
    assert kit.started[-1] == (
        agent.id,
        "A reviewer checked your work and it did not pass:\nFAIL\ntest_discount still fails\n"
        "Fix this, then report again.",
    )
    await kit.runner.finished(agent.id, ended("ok", "Fixed the discount."))
    second = await reviewer_of(kit, agent)
    assert second.id != first.id
    await kit.runner.finished(second.id, ended("ok", "FAIL\nstill broken"))
    await settle(kit)
    done = await kit.agents.get(agent.id)
    assert done is not None and done.status == "done" and done.reviews == VERIFY_ROUNDS
    text = kit.notices[-1][1]
    assert f"did not pass {VERIFY_ROUNDS} reviews" in text and "still broken" in text
    assert "Fixed the discount." in text


async def test_reviewers_do_not_count_against_the_limit(kit: RunnerKit) -> None:
    _, agent = await checked_agent(kit)
    await kit.runner.finished(agent.id, ended("ok", "Done."))
    reviewer = await reviewer_of(kit, agent)
    await kit.runner.finished(reviewer.id, ended("error", error="boom"))
    await settle(kit)
    assert "could not be reviewed" in kit.notices[-1][1]
    other = await kit.sessions.create()
    started = await kit.runner.start("next", "More work", other.id)
    assert started.status == "working"


async def test_an_agent_notes_its_progress(kit: RunnerKit) -> None:
    _, agent = await checked_agent(kit)
    assert agent.session_id is not None
    updated = await kit.runner.note_progress(agent.session_id, " Wrote the parser; tests next. ")
    assert updated.progress == "Wrote the parser; tests next."
    assert (await kit.agents.get(agent.id)).progress == updated.progress  # type: ignore[union-attr]


async def test_waiting_takes_the_report_so_no_message_follows(kit: RunnerKit) -> None:
    parent = await kit.sessions.create()
    agent = await kit.runner.start("reader", "Read", parent.id)
    waiter = asyncio.create_task(kit.runner.wait(parent.id, [], 5, CancelToken()))
    await settle()
    await kit.runner.finished(agent.id, ended("ok", "Three files."))
    found = await waiter
    assert found.messages == [
        f"Agent 'reader' (id {agent.id}) has finished. Its report:\nThree files."
    ]
    await settle()
    assert kit.notices == []


async def test_waiting_ends_when_nobody_works_or_time_runs_out(kit: RunnerKit) -> None:
    parent = await kit.sessions.create()
    nobody = await kit.runner.wait(parent.id, [], 5, CancelToken())
    assert nobody.messages == [] and nobody.working == []
    agent = await kit.runner.start("slow", "Take long", parent.id)
    late = await kit.runner.wait(parent.id, [agent.id], 0.05, CancelToken())
    assert late.messages == [] and late.working == [f"Agent 'slow' (id {agent.id})"]
    cancel = CancelToken()
    cancel.cancel()
    stopped = await kit.runner.wait(parent.id, [], 5, cancel)
    assert stopped.messages == []
