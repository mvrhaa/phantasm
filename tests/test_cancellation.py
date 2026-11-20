"""Tests for phantasm.core.cancellation."""

import asyncio

import pytest

from phantasm.core.cancellation import CancelToken, TurnCancelled
from phantasm.providers.base import CancelSignal


def test_new_token_is_not_cancelled() -> None:
    token = CancelToken()

    assert token.cancelled is False
    token.raise_if_cancelled()


def test_cancel_sets_the_flag_for_good() -> None:
    token = CancelToken()

    token.cancel()
    token.cancel()

    assert token.cancelled is True


def test_raise_if_cancelled_raises_after_cancel() -> None:
    token = CancelToken()
    token.cancel()

    with pytest.raises(TurnCancelled):
        token.raise_if_cancelled()


def test_token_can_be_created_outside_an_event_loop() -> None:
    token = CancelToken()

    async def use() -> bool:
        token.cancel()
        await token.wait()
        return token.cancelled

    assert asyncio.run(use()) is True


async def test_wait_blocks_until_cancel() -> None:
    token = CancelToken()
    waiter = asyncio.create_task(token.wait())

    await asyncio.sleep(0)
    assert not waiter.done()

    token.cancel()
    await asyncio.wait_for(waiter, timeout=1)
    assert waiter.done()


async def test_wait_returns_at_once_when_already_cancelled() -> None:
    token = CancelToken()
    token.cancel()

    await asyncio.wait_for(token.wait(), timeout=1)


async def test_cancel_wakes_every_waiter() -> None:
    token = CancelToken()
    waiters = [asyncio.create_task(token.wait()) for _ in range(3)]
    await asyncio.sleep(0)

    token.cancel()

    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1)


async def test_wait_can_race_against_other_work() -> None:
    token = CancelToken()
    work = asyncio.create_task(asyncio.sleep(60))
    cancelled = asyncio.create_task(token.wait())

    token.cancel()
    done, pending = await asyncio.wait({work, cancelled}, return_when=asyncio.FIRST_COMPLETED)

    assert done == {cancelled}
    assert pending == {work}
    work.cancel()


def test_tokens_are_independent() -> None:
    first, second = CancelToken(), CancelToken()

    first.cancel()

    assert second.cancelled is False


def test_token_satisfies_the_provider_cancel_signal() -> None:
    def accepts(signal: CancelSignal) -> bool:
        return signal.cancelled

    assert accepts(CancelToken()) is False
